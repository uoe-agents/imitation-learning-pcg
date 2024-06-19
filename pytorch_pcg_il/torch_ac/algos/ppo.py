import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, env_type, envs, envs_evaluation, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.0001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence_actor=False, recurrence_critic=False,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, num_actions = 7,
                 separated_networks=1, env_name='MiniGrid-DoorKey-5x5-v0',
                 int_coef=0.0, normalize_int_rewards = 0,
                 im_type = 'counts', 
                 use_episodic_counts=0, use_only_not_visited=0,total_num_frames=20000000,
                 sl_clipgrad=0,nminibatch=4,num_train_seeds=-1,init_level_seed=0,
                  ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(env_type, envs,envs_evaluation, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence_actor, recurrence_critic, preprocess_obss, reshape_reward,
                         separated_networks, env_name, num_actions, int_coef, normalize_int_rewards,
                         im_type,use_episodic_counts,use_only_not_visited,total_num_frames,
                         num_train_seeds,init_level_seed)

        # Imitation Learning params
        self.num_actions = num_actions
        self.sl_clipgrad = sl_clipgrad

        # PPO related params
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.use_value_clipping = True #compute critic loss with MSE
        
        # self.separated_actor_critic = separated_networks for actor and critic
        if self.separated_actor_critic:
            self.optimizer = [torch.optim.Adam(self.acmodel[i].parameters(), lr, eps=adam_eps)
                                for i in range(len(self.acmodel))]  
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)

        # losses
        self.forward_mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        # Mini-Batches related
        self.batch_size = batch_size # deprecated with the use of minibatches
        self.batch_num = 0
        self.nminibatch = nminibatch
        self.rollout_length = len(envs)*num_frames_per_proc
        # Check incompatibilities
        assert self.rollout_length % self.nminibatch == 0
        self.minibatch_size = self.rollout_length // self.nminibatch
        print('\n{} Minibatches of size size:{}\n'.format(self.nminibatch,self.minibatch_size))

    def update_parameters_imitation(self,lr,obs,acs,dones,next_obs,values_il_target=[]):
        if True:
            return self.update_parameters_imitation_bc(lr,obs,acs,dones,next_obs,values_il_target)
        # possibility to add more types of imitation learning losses or approaches here

    def update_parameters_imitation_bc(self,lr,obs,acs,dones,next_obs,values_il_target=[]):
        obs = torch.from_numpy(obs).to(self.device).float()
        acs = torch.from_numpy(acs).to(self.device)

        ####################################################
        # FORWARD PASS
        ####################################################
        if self.separated_actor_critic:
            # ACTOR
            dist, logp_soft, _ = self.acmodel[0](obs) #actor
            # CRITIC RELATED
            if len(values_il_target)>0:
                value_ext = self.acmodel[1](obs) #critic

        else:
            dist, logp_soft, value_ext = self.acmodel(obs)

        ####################################################
        # NLL LOSS (ONLY REQUIRED FOR POLICY)
        ####################################################
        log_soft_pred_actions = logp_soft # log_softmax logits
        target_actions = acs.long() # logits, need to be long (if not --> RuntimeError: expected scalar type Double but found Float)

        # compute loss
        rapid_loss = F.nll_loss(log_soft_pred_actions,
                                target = target_actions,
                                reduction='mean')        
        # print('rapid loss shape:', rapid_loss.shape)

        # with cross entropy (one-hot version; not necessary, internally computed)
        # actions_one_hot = torch.zeros((acs.shape[0],self.num_actions),device=self.device)
        # for i,a in enumerate(acs):
        #     one_hot = F.one_hot(a, num_classes=self.num_actions).float()
        #     actions_one_hot[i].copy_(one_hot)
        # rapid_crossentropy_loss = self.cross_entropy(logp_soft,actions_one_hot)
        # print('rapid cross_entropy:',rapid_crossentropy_loss)

        # with cross entropy (no log_soft)
        # actions_one_hot = torch.zeros((acs.shape[0],self.num_actions),device=self.device)
        # rapid_crossentropy_loss = self.cross_entropy(logp_soft,acs)
        # print('rapid cross_entropy:',rapid_crossentropy_loss)

        ####################################################
        # BACKPROPAGATION
        ####################################################
        if self.separated_actor_critic:
            # critic related
            if len(values_il_target)>0:
                values_il_target = torch.from_numpy(values_il_target).to(self.device).float()
                value_loss = self.forward_mse(value_ext,values_il_target).mean()
                self.optimizer[1].zero_grad()
                value_loss.backward()
                self.optimizer[1].step()

            # actor related
            self.optimizer[0].zero_grad()
            rapid_loss.backward()
            # gradient clipping
            if self.sl_clipgrad:
                torch.nn.utils.clip_grad_norm_(self.acmodel[0].parameters(), self.max_grad_norm)
            grad_norm = self.calculate_gradients(self.acmodel[0])
            self.optimizer[0].step()
        else:
            # critic related
            if len(values_il_target)>0:
                values_il_target = torch.from_numpy(values_il_target).to(self.device).float()
                value_loss = self.forward_mse(value_ext,values_il_target).mean()
                # both losses combined
                rapid_loss = rapid_loss + value_loss

            # apply single backpropagation step
            self.optimizer.zero_grad()
            rapid_loss.backward()
            # gradient clipping
            if self.sl_clipgrad:
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            grad_norm = self.calculate_gradients(self.acmodel)
            self.optimizer.step()

        # return monitorization parameters
        return rapid_loss.item(), grad_norm

    def update_parameters_ppo(self, exps):

        length_rollout = exps.action.shape[0]
        assert self.rollout_length == length_rollout

        idx = np.arange(length_rollout)
        # select size of minibatch
        self.minibatch_size = length_rollout // self.nminibatch
        # Collect experiences
        for _ in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            #gradients
            log_grad_norms = []
            log_grad_norms_critic = []
            # losses
            log_policy_losses = []
            log_value_losses = []

            # shuffle
            np.random.shuffle(idx)

            for start in range(0,length_rollout,self.minibatch_size):
                # select idx
                end = start + self.minibatch_size
                mb_idx = idx[start:end]

                # Initialize batch values
                # policy
                batch_entropy = 0
                batch_policy_loss = 0
                # critic
                batch_value = 0
                batch_value_loss = 0
                # both
                batch_loss = 0

                for _ in range(1): #for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[mb_idx]

                    # Compute loss
                    if self.separated_actor_critic:
                        dist, _, _ = self.acmodel[0](obs=sb.obs) #actor
                        value_ext = self.acmodel[1](obs=sb.obs) #critic
                    else:
                        dist, _, value_ext = self.acmodel(obs=sb.obs)


                    # Advs normalization
                    sb.advantage_total = (sb.advantage_total - sb.advantage_total.mean()) / (sb.advantage_total.std() + 1e-8)

                    # Policy related
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage_total
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage_total
                    policy_loss = -torch.min(surr1, surr2).mean()

                    if self.use_value_clipping:
                        # only extrinsic head
                        value_clipped = sb.value_ext + torch.clamp(value_ext - sb.value_ext, -self.clip_eps, self.clip_eps)
                        surr1 = self.forward_mse(value_ext,sb.returnn_ext)
                        surr2 = self.forward_mse(value_clipped,sb.returnn_ext)
                        value_loss = torch.max(surr1, surr2).mean()
                    else:
                        value_loss = self.forward_mse(value_ext,sb.returnn_ext)
                    # self.forward_mse

                    # total loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values
                    #policy
                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    #critic
                    batch_value += value_ext.mean().item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                # Update actor-critic
                if self.separated_actor_critic:

                    self.optimizer[0].zero_grad()
                    self.optimizer[1].zero_grad()
                    batch_loss.backward()
                    # grad_norm_actor_before = self.calculate_gradients(self.acmodel[0])
                    # grad_norm_critic_before = self.calculate_gradients(self.acmodel[1])
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.acmodel[0].parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.acmodel[1].parameters(), self.max_grad_norm)
                    grad_norm = grad_norm_actor = self.calculate_gradients(self.acmodel[0])
                    grad_norm_critic = self.calculate_gradients(self.acmodel[1])
                    self.optimizer[0].step()
                    self.optimizer[1].step()
                else:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    # grad_norm_before = self.calculate_gradients(self.acmodel)

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    grad_norm = self.calculate_gradients(self.acmodel)
                    grad_norm_critic = 0
                    self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_policy_losses.append(batch_policy_loss)

                log_values.append(batch_value)
                log_value_losses.append(batch_value_loss)

                log_grad_norms.append(grad_norm) #used to monitore either AC or just the actor
                log_grad_norms_critic.append(grad_norm_critic) # when having 2 networks, we monitore the critic here

                # *** ONE BACKPROPAGATION DONE WITH ONE MINIBATCH ***
            # ***ONE EPOCH FINISHED, AFTER ALL THE GRADIENTS OF ALL MINIBATCHES
        # *** ALL EPOCHS FINISHED ***
        # Log some values
        logs = {
            "entropy": np.mean(log_entropies),
            "policy_loss": np.mean(log_policy_losses),
            "value": np.mean(log_values),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms),
            "grad_norm_critic": np.mean(log_grad_norms_critic)
        }

        return logs

    def calculate_gradients(self, model):
        # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5 #monitorization of gradients
        grad_norm = 0
        for enum,p in enumerate(model.parameters()):
            # print('{} Parameter shape: {}'.format(enum,p.shape))
            # print('Grad:',p.grad)
            try:
                grad_norm += p.grad.data.norm(2).item() ** 2
            except AttributeError:
                # print('No grad')
                continue
        grad_norm = grad_norm **0.5

        return grad_norm
