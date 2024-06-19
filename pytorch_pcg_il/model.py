import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ActorModel_RAPID(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()
        self.use_memory = use_memory
        activ_func = nn.Tanh()
        # activ_func = nn.ReLU()
        input_size=obs_space#7*7*3

        # init
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                    constant_(x, 0), nn.init.calculate_gain('relu'))

        if self.use_memory:
            self.fc = nn.Sequential(init_(nn.Linear(input_size,64)),activ_func)
            self.memory_rnn = nn.LSTMCell(64, self.semi_memory_size)
            self.output_lstm_fc = self.semi_memory_size
        else:
            self.fc = nn.Sequential(
                init_(nn.Linear(input_size,64)),
                activ_func,
                init_(nn.Linear(64,64)),
                activ_func
            )
            self.output_lstm_fc = 64

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        # Define actor's model
        self.actor = nn.Sequential(init_(nn.Linear(self.output_lstm_fc, action_space)))
    @property
    def memory_size(self):
        return 2*self.semi_memory_size
    @property
    def semi_memory_size(self):
        return 64

    def forward(self, obs, memory=[]):
        try:
            obs_flatened = obs.image.view(obs.image.shape[0], -1)
        except AttributeError:
            obs_flatened = obs.view(obs.shape[0], -1)

        if self.use_memory:
            x = self.fc(obs_flatened)
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            memory = torch.cat(hidden, dim=1)
            embedding = hidden[0]
        else:
            embedding = self.fc(obs_flatened)

        x = self.actor(embedding)
        policy_logits = F.log_softmax(x, dim=1)
        dist = Categorical(logits=policy_logits)

        if self.use_memory:
            return dist, policy_logits, embedding , memory
        else:
            return dist, policy_logits, embedding
class CriticModel_RAPID(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        self.use_memory = use_memory
        activ_func = nn.Tanh()
        # activ_func = nn.ReLU()
        input_size=obs_space#7*7*3

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        if self.use_memory:
            self.fc = nn.Sequential(init_(nn.Linear(input_size,64)),activ_func)
            self.memory_rnn = nn.LSTMCell(64, self.semi_memory_size)
            self.output_lstm_fc = self.semi_memory_size
        else:
            self.fc = nn.Sequential(
                init_(nn.Linear(input_size,64)),
                activ_func,
                init_(nn.Linear(64,64)),
                activ_func
            )
            self.output_lstm_fc = 64

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_ext = nn.Sequential(init_(nn.Linear(self.output_lstm_fc, 1)))
    @property
    def memory_size(self):
        return 2*self.semi_memory_size
    @property
    def semi_memory_size(self):
        return 64

    def forward(self, obs, memory=[]):
        try:
            obs_flatened = obs.image.view(obs.image.shape[0], -1)
        except AttributeError:
            obs_flatened = obs.view(obs.shape[0], -1)

        if self.use_memory:
            x = self.fc(obs_flatened)
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            memory = torch.cat(hidden, dim=1)
            embedding = hidden[0]
        else:
            embedding = self.fc(obs_flatened)

        x = self.critic_ext(embedding)
        value_ext = x.squeeze(1)

        if self.use_memory:
            return value_ext, memory
        else:
            return value_ext
class ACModel_LSTM(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define image embedding
        self.image_conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
        )
        # n,m = obs_space["image"][0],obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_embedding_size = 32

        # Define memory and embedding size based of we use FC or LSTM
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            self.output_lstm_fc = self.semi_memory_size# ensure memory size is accordingly to next actor/critic neurons
            # self.fc = nn.Sequential(init_(nn.Linear(256,256)),nn.ReLU())
            # self.output_lstm_fc = 256# ensure memory size is accordingly to next actor/critic neurons
        else:
            self.fc = nn.Sequential(init_(nn.Linear(self.image_embedding_size,256)),nn.ReLU())
            self.output_lstm_fc = 256# ensure memory size is accordingly to next actor/critic neurons

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        # Define actor's model
        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_lstm_fc, action_space)),
        )

        # Define critic's model
        self.critic_ext = nn.Sequential(
            init_(nn.Linear(self.output_lstm_fc, 1))
        )

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return 64#self.image_embedding_size

    def forward(self, obs, memory=[]):
        try:
            x = obs.image.transpose(1, 3).transpose(2, 3)
        except AttributeError:
            x = obs.transpose(1, 3).transpose(2, 3)

        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            # embedding = self.fc(embedding)
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = self.fc(x)

        x = self.actor(embedding)
        policy_logits = F.log_softmax(x, dim=1)
        dist = Categorical(logits=policy_logits)

        x = self.critic_ext(embedding)
        value_ext = x.squeeze(1)


        if len(memory)>0:
            return dist, policy_logits, value_ext, memory
        else:
            return dist, policy_logits, value_ext

import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ACModel_PROCGEN(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        h, w, c = obs_space
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, action_space))
        self.critic = layer_init(nn.Linear(256, 1))

    def forward(self,obs):
        try:
            obs = obs.image
        except AttributeError:
            obs = obs

        embedding = self.network(obs.permute((0, 3, 1, 2)) / 255.0) # "bhwc" -> "bchw"
        # critic
        value = self.critic(embedding)
        value = value.squeeze(1)

        # actor
        logits = self.actor(embedding)
        policy_logits = F.log_softmax(logits,dim=1)
        dist = Categorical(logits=policy_logits)

        # return values
        return dist, policy_logits, value