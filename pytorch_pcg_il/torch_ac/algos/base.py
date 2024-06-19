from abc import ABC, abstractmethod
import torch
import numpy as np
from copy import deepcopy
from collections import deque
from statistics import mean

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch_ac.utils import RunningMeanStd
from torch_ac.utils.intrinsic_motivation import CountModule,BeBold

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, env_type, envs, envs_evaluation, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence_actor, recurrence_critic, preprocess_obss, reshape_reward,
                 separated_ac_networks, env_name, num_actions,
                 int_coef, normalize_int_rewards,
                 im_type, use_episodic_counts,use_only_not_visited,
                 total_num_frames,num_train_seeds,init_level_seed):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module or tuple of torch.Module(s)
            the model(s); the separated_actor_critic parameter defines that
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        separated_networks: boolean
            set whether we are going to use a single AC neural network or
            two differents
        """
        self.env_type = env_type # 'procgen or minigrid
        # when to activate episode limitation --> used to delimit the number of maximum episodes/level within a single rollout
        self.episode_limitation = False

        # Store parameters
        self.separated_actor_critic = separated_ac_networks
        self.acmodel = acmodel

        self.num_actions = num_actions
        self.device = device

        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Configure acmodel

        if self.separated_actor_critic:
            for i in range(len(self.acmodel)):
                self.acmodel[i].to(self.device)
                self.acmodel[i].train()
        else:
            self.acmodel.to(self.device)
            self.acmodel.train()

        # environment
        self.env = ParallelEnv(envs,env_name)
        
        if self.env_type == 'procgen':
            self.ev_env = ParallelEnv(envs_evaluation,env_name) # just for evaluation (procgen)
        else:
            self.ev_env = ParallelEnv(envs,env_name) # just for evaluation (minigrid)
        
        self.num_frames_per_proc = num_frames_per_proc
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.total_num_frames = total_num_frames
        print('total frames:',self.total_num_frames)

        shape = (self.num_frames_per_proc, self.num_procs)

        # recurrence
        self.use_recurrence_actor = recurrence_actor
        self.use_recurrence_critic = recurrence_critic
        self.use_recurrence = True if recurrence_actor or recurrence_critic else False

        if self.separated_actor_critic:
            if self.use_recurrence_actor:
                # actor params
                memory_size_actor = self.acmodel[0].memory_size # actor
                self.memory_actor = torch.zeros(shape[1], memory_size_actor, device=self.device) # hidden state in a single env step for each parallel agent
                self.memories_actor = torch.zeros(*shape, memory_size_actor, device=self.device) # hidden states in the whole trajectory for each parallel agent
            if self.use_recurrence_critic:
                # critic params
                memory_size_critic = self.acmodel[1].memory_size # critic
                self.memory_critic = torch.zeros(shape[1], memory_size_critic, device=self.device) # hidden state in a single env step for each parallel agent
                self.memories_critic = torch.zeros(*shape, memory_size_critic, device=self.device) # hidden states in the whole trajectory for each parallel agent
        else:
            if self.use_recurrence:
                # single AC network
                memory_size = self.acmodel.memory_size # ac
                self.memory = torch.zeros(shape[1], memory_size, device=self.device) # hidden state in a single env step for each parallel agent
                self.memories = torch.zeros(*shape, memory_size, device=self.device) # hidden states in the whole trajectory for each parallel agent

        # Store helpers values
        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values_ext = torch.zeros(*shape, device=self.device)
        self.advantages_ext = torch.zeros(*shape, device=self.device)
        self.returns_ext = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        self.num_train_seeds = num_train_seeds
        self.init_level_seed = init_level_seed
        if self.env_type == 'procgen':
            self.specify_seeds = False
        else:
            self.specify_seeds = True if self.num_train_seeds > 0 else False
            
        
        self.seed = np.ones(shape[1], dtype=int) # current episode seed
        self.seeds = np.ones(shape[1], dtype=int) # next episodes seed

        # Initialize LOGs values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device) # monitores the return inside the episode (it increases with each step until done is reached)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs # monitores the total return that was given in the whole episode (updates after each episode)
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        self.episode_counter = 0
        self.frames_counter = 0

        # for intrinsic coef adaptive decay
        self.log_rollout_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_train = torch.tensor([],device=self.device) # stores the avg return reported after each rollout (avg of all penv after every nsteps)

        # *****  Intrinsic motivation related parameters *****
        self.int_coef = int_coef
        self.use_normalization_intrinsic_rewards = normalize_int_rewards
        self.im_type = im_type 
        # define IM module
        if self.im_type == 'counts':
            print('\nUsing COUNTS')
            self.im_module = CountModule()

        elif self.im_type == 'bebold':
            print('\nUsing BEBOLD')
            self.im_module = BeBold()

        # episodic counts and first visit variables
        self.use_episodic_counts = 1 if im_type == 'ride' else use_episodic_counts # ride always uses episodic counts by default
        self.episodic_counts = [CountModule() for _ in range(self.num_procs)] # counts used to carry out how many times each observation has been visited inside an episode
        self.use_only_not_visited = use_only_not_visited
        self.visited_state_in_episode = torch.zeros(*shape, device=self.device) # mask that is used to allow or not compute a non-zero intrinsic reward

        # Parameters needed when using two-value/advantage combination for normalization
        self.return_rms = RunningMeanStd()
        self.normalization_int_score = 0
        self.min_std = 0.01
        self.predominance_ext_over_int = torch.zeros(*shape, device=self.device)

        # experience values
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.rewards_total = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)
        self.advantages_total = torch.zeros(*shape, device=self.device)
        self.returns_int = torch.zeros(*shape, device=self.device)
        # add monitorization for intrinsic part
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
        # other for normalization
        self.log_episode_return_int_normalized = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_normalized =  [0] * self.num_procs
        # add avg 100 episodes return
        self.last_100success = deque([0],maxlen=100)
        self.last_100return = deque([0],maxlen=100)
        self.last_100steps = deque([0],maxlen=100)
        self.last_100return_int = deque([0],maxlen=100)

        # ***EPISODE MANAGEMENT
        self.episode_info = [] # a list that contains dicts inside: obs, acs and ret as keys
        self.episode_obs = [[] for _ in range(self.num_procs)]
        self.episode_acs = [[] for _ in range(self.num_procs)]
        self.episode_rew = [[] for _ in range(self.num_procs)]
        self.episode_nobs = [[] for _ in range(self.num_procs)]
        self.episode_dones = [[] for _ in range(self.num_procs)]
        self.episode_values = [[] for _ in range(self.num_procs)]
        self.episode_advantages = [[] for _ in range(self.num_procs)]

        self.episode_mem = [[] for _ in range(self.num_procs)]

        # To see the display Window
        from gym_minigrid.window import Window
        self.window_visible = False
        self.window = Window('gym_minigrid - ')

        print('num_frame per proc:',self.num_frames_per_proc)
        print('num of process:',self.num_procs)
        print('num frames (num_pallel envs*framesperproc):', self.num_frames)

    def reset_episode_info(self):
        # reset episode_info --> called from train.py
        self.episode_info = []

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        num_episodes = 0
        new_num_frames_per_proc = 0
        
        for i in range(self.num_frames_per_proc):

            if self.window_visible:
                self.window.set_caption('Episode: {}, step: {}'.format(self.episode_counter,i))

            # update frame counter after each step
            self.frames_counter += self.num_procs

            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.separated_actor_critic:
                    dist, logits, _embedding, = self.acmodel[0](preprocessed_obs)
                    value = self.acmodel[1](preprocessed_obs)
                else:
                    dist, logits, value = self.acmodel(obs=preprocessed_obs)

            # take action from distribution
            action = dist.sample()

            # *** ENVIRONMENT STEP ***
            if self.specify_seeds:
                obs, reward, done, info = self.env.step(actions=action.cpu().numpy(),seeds=self.seeds)
            else:
                obs, reward, done, info = self.env.step(actions=action.cpu().numpy(),seeds=len(self.seeds)*[None])

            # to see window (interactive)
            if self.window_visible:
                img = self.env.render('rgb_array')
                self.window.show_img(img)

            # Update experience values (for episode granularity)
            obs_np = preprocessed_obs.image.cpu().numpy()
            acs_np = action.cpu().numpy()
            rew_np = np.array(reward)
            nobs_np = self.preprocess_obss(obs, device=self.device).image.cpu().numpy()
            done_np = np.array(done)
            val_np = value.cpu().numpy()

            # if self.use_recurrence:
            #     mem_np = self.memory.cpu().numpy()

            for aidx in range(self.num_procs):
                
                # add experience to be stored in the buffer
                self.episode_obs[aidx].append(obs_np[aidx])
                self.episode_acs[aidx].append(acs_np[aidx])
                self.episode_rew[aidx].append(rew_np[aidx])
                self.episode_nobs[aidx].append(nobs_np[aidx])
                self.episode_dones[aidx].append(done_np[aidx])
                self.episode_values[aidx].append(val_np[aidx])
                
                # if self.use_recurrence:
                #     self.episode_mem[aidx].append(mem_np[aidx])

                if np.array(done)[aidx] == True:
                    num_episodes += 1
                    
                    aux_dict = {}
                    aux_dict['obs'] = np.array(self.episode_obs[aidx])
                    aux_dict['acs'] = np.array(self.episode_acs[aidx])
                    aux_dict['rew'] = np.array(self.episode_rew[aidx]) # reward in each step
                    aux_dict['nobs'] = np.array(self.episode_nobs[aidx]) # s'
                    aux_dict['dones'] = np.array(self.episode_dones[aidx]) # dones
                    aux_dict['ret'] = np.sum(self.episode_rew[aidx]) # non-discounted return of episode
                    aux_dict['steps'] = steps_in_episode = len(self.episode_rew[aidx])
                    aux_dict['seed'] = info[0]['prev_level_seed'] if self.env_type=='procgen' else self.seed[aidx]

                    

                    # CALCULATE ADVANTAGES/GAE -- As Regret
                    self.episode_advantages[aidx] = np.zeros(steps_in_episode)
                    for jj in reversed(range(steps_in_episode)):
                        next_mask = 1 - np.array(self.episode_dones[aidx][jj]) if jj < steps_in_episode - 1 else 1 - np.array(self.episode_dones[aidx][jj],dtype=int)
                        next_value = self.episode_values[aidx][jj+1] if jj < steps_in_episode - 1 else 0
                        next_advantage = self.episode_advantages[aidx][jj+1] if jj < steps_in_episode - 1 else 0

                        delta = self.episode_rew[aidx][jj] + self.discount * next_value * next_mask - self.episode_values[aidx][jj]
                        self.episode_advantages[aidx][jj] = max(delta + self.discount * self.gae_lambda * next_advantage * next_mask, 0)

                    gae_score = np.mean(self.episode_advantages[aidx])
                    aux_dict['gae_score'] = gae_score

                    # if self.use_recurrence:
                    #     aux_dict['mem'] = np.array(self.episode_mem[aidx])

                    # add dictionary to list containing other episodes
                    self.episode_info.append(aux_dict)
                    
                    # reset lists
                    self.episode_obs[aidx] = []
                    self.episode_acs[aidx] = []
                    self.episode_rew[aidx] = []
                    self.episode_nobs[aidx] = []
                    self.episode_dones[aidx] = []
                    self.episode_values[aidx] = []
                    self.episode_advantages[aidx] = []
                    self.episode_mem[aidx] = []

            ####################################################################
            # Update experiences values
            ####################################################################

            self.obss[i] = self.obs # stores the current observation on the experience
            self.obs = obs # stores the next_obs obtained after the step in the env
            self.actions[i] = action
            self.values_ext[i] = value

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            """
            # UPDATE RECURRENCY RELATED
            if self.separated_actor_critic:
                if self.use_recurrence_actor:
                    self.memories_actor[i] = self.memory_actor
                    self.memory_actor = memory_actor * self.mask.unsqueeze(1)
                if self.use_recurrence_critic:
                    self.memories_critic[i] = self.memory_critic
                    self.memory_critic = memory_critic * self.mask.unsqueeze(1)
            else:
                if self.use_recurrence:
                    self.memories[i] = self.memory
                    self.memory = memory * self.mask.unsqueeze(1)
            """
            
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)


            ####################################################################
            # ***Intrinsic motivation - calculate intrinsic rewards
            ####################################################################
            if self.int_coef > 0:

                # calculate bonus (step by step)  - shape [num_procs, 7,7,3]
                input_current_obs = preprocessed_obs
                input_next_obs = self.preprocess_obss(self.obs, device=self.device) # contains next_observations
                # FOR COMPUTING INTRINSIC REWARD, THE REQUIRED SHAPE IS JUST A UNIT -- i.e image of [7,7,3]; action shape of [1] (it is calculated one by one)
                # FOR UPDATING COUNTS (done IN BATCH for efficiency), the shape requires to have the batch-- i.e image of [batch,7,7,3]; action of [batch,1]
                rewards_int = [self.im_module.compute_intrinsic_reward(obs=ob,next_obs=nobs,actions=act) \
                                            for ob,nobs,act in zip(input_current_obs.image, input_next_obs.image, action)]

                self.rewards_int[i] = rewards_int_torch = torch.tensor(rewards_int,device=self.device,dtype=torch.float)

                # Scale rewards
                if self.im_type=='bebold' or self.use_episodic_counts or self.use_only_not_visited:
                    # Update episodic counter (mandatory for both episodic and 1st visitation count strategies)
                    num_episodic_counts = np.zeros(self.num_procs)
                    # we need to squeeze to have actions of shape [num_procs, 1, 1] and also the observations [num_procs,1,7,7,3]
                    for penv,(ob,nobs,act) in enumerate(zip(input_current_obs.image.unsqueeze(1), input_next_obs.image.unsqueeze(1) , action.unsqueeze(1).unsqueeze(1) ) ):
                        self.episodic_counts[penv].update(obs=ob,next_obs=nobs,actions=act)
                        # get number of times visited in episode current step (is based on s'!)
                        num_episodic_counts[penv] = self.episodic_counts[penv].get_number_visits(obs=nobs, actions=act)

                    # Reward only when agent visits state s for the first time in the episode
                    if self.im_type=='bebold' or self.use_only_not_visited:
                        # check if state has already been visited -- mask (1 means it has been just visited once)
                        visit_erir = [1 if numv == 1 else 0 for numv in num_episodic_counts]
                        self.rewards_int[i] = rewards_int_torch * torch.tensor(visit_erir,device=self.device,dtype=torch.int)

                    else: #self.use_episodic_counts:
                        episodic_reward_factor = 1/np.sqrt(num_episodic_counts)
                        self.rewards_int[i] = rewards_int_torch * torch.from_numpy(episodic_reward_factor).to(self.device)


            ####################################################################
            # Update log values
            ####################################################################
            if self.int_coef > 0:
                self.log_episode_return_int += self.rewards_int[i]
                self.log_episode_return_int_normalized += torch.tensor(np.asarray(rewards_int)/max(self.normalization_int_score,self.min_std), device=self.device, dtype=torch.float)
                # used for adaptive int coef
                self.log_rollout_return_int += self.rewards_int[i]
                # print('log episode_return:',self.log_episode_return_int)
                # print('log rollout return',self.log_rollout_return_int)

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            # for all the num_procs...
            for i, done_ in enumerate(done):
                if done_:
                    self.episodic_counts[i].reset()

                    # reset seed
                    if self.specify_seeds:
                        self.seed[i] = self.seeds[i]
                        self.seeds[i] = np.random.randint(low=self.init_level_seed,
                                                          high=self.init_level_seed + self.num_train_seeds) #used to determine next sampled level

                    # log related
                    self.log_done_counter += 1
                    self.episode_counter += 1

                    if self.int_coef > 0:
                        self.log_return_int.append(self.log_episode_return_int[i].item())
                        self.log_return_int_normalized.append(self.log_episode_return_int_normalized[i].item())
                        self.last_100return_int.append(self.log_episode_return_int[i].item())

                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    # moving average of 100 episodes
                    self.last_100success.append(1 if self.log_episode_return[i].item() > 0 else 0)
                    self.last_100return.append(self.log_episode_return[i].item())
                    self.last_100steps.append(self.log_episode_num_frames[i].item())


            # Reset values if the process finished the episode (mask!=done)
            if self.int_coef > 0:
                self.log_episode_return_int *= self.mask
                self.log_episode_return_int_normalized *= self.mask

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

            # update the counter
            new_num_frames_per_proc += 1
            if (num_episodes >= 8) and (self.episode_limitation==True): #currently not working; used for testing
                break

            # **********************************************************************
            # ONE STEP INSIDE THE ROLLOUT COMPLETED
            # **********************************************************************
        # **********************************************************************
        # ROLLOUT COLLECTION FINISHED.
        # **********************************************************************

        # 1.Update IM Module
        # 2.Normalize intrinsic rewards (before training)

        # Part 1 of updating...
        if self.int_coef > 0:
            # 1.1. preprocess the batch of data to be Tensors
            shape_im = (self.num_frames_per_proc,self.num_procs, 7,7,3) # preprocess batch observations (num_steps*num_instances, 7 x 7 x 3)
            input_obss = torch.zeros(*shape_im,device=self.device)
            input_nobss = torch.zeros(*shape_im,device=self.device)

            # generate next_states (same as self.obss + an additional next_state of al the penvs)
            nobss = deepcopy(self.obss)
            nobss = nobss[1:] # pop first element and move left
            nobss.append(self.obs) # add at the last position the next_states

            for num_frame,(mult_obs,mult_nobs) in enumerate(zip(self.obss,nobss)): # len(self.obss) ==> num_frames_per_proc == number_of_step

                for num_process,(obss,nobss) in enumerate(zip(mult_obs,mult_nobs)):
                    o = torch.tensor(obss['image'], device=self.device)
                    no = torch.tensor(nobss['image'], device=self.device)
                    input_obss[num_frame,num_process].copy_(o)
                    input_nobss[num_frame,num_process].copy_(no)

            # 1.2. reshape to have [num_frames*num_procs, 7, 7, 3]
            input_obss = input_obss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_nobss = input_nobss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_actions = self.actions.view(self.num_frames_per_proc*self.num_procs,-1)

            # print('\nIM:')
            # self.im_module.visualize_counts()

            # 1.3. Update
            self.im_module.update(obs=input_obss,next_obs=input_nobss,actions=input_actions)

            # 2. Normalize (if required)
            if self.use_normalization_intrinsic_rewards:
                # Calculate normalization after each rollout
                batch_mean, batch_var, batch_count = self.log_rollout_return_int.mean(-1).item(), self.log_rollout_return_int.var(-1).item(), len(self.log_rollout_return_int)
                self.return_rms.update_from_moments(batch_mean, batch_var, batch_count)
                self.normalization_int_score = np.sqrt(self.return_rms.var)
                self.normalization_int_score = max(self.normalization_int_score, self.min_std)

                # apply normalization
                self.rewards_int /= self.normalization_int_score

        # **********************************************************************
        # obtain next_value for computing advantages and return
        # **********************************************************************
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.separated_actor_critic:
                next_value = self.acmodel[1](obs=preprocessed_obs)
            else:
                _, _, next_value = self.acmodel(obs=preprocessed_obs)

        # **********************************************************************
        # ***Combining EXTRINSIC-INTRINSIC rewards***
        # **********************************************************************
        # 1. Obtain the total reward
        if self.int_coef <= 0:
            # No intrinsic_rewards
            self.rewards_total.copy_(self.rewards)
        else:
            # Using Intrinsic Motivation
            self.rewards_total = self.rewards + self.int_coef*self.rewards_int
            self.rewards_total /= (1+self.int_coef)

        # 2. Calculate advantages and returns
        for i in reversed(range(new_num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < new_num_frames_per_proc - 1 else self.mask
            next_value = self.values_ext[i+1] if i < new_num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages_ext[i+1] if i < new_num_frames_per_proc - 1 else 0

            delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values_ext[i]
            self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        self.returns_ext =  self.values_ext + self.advantages_ext
        # print('\n**RETURNS: ',self.returns_ext)
        # print('\n**ADVANTAGES: ',self.advantages_ext)
        self.advantages_total.copy_(self.advantages_ext)

        ########################################################################
        # *** Re-fill up to T-size with previous obtained samples (uniformly ) ***
        ########################################################################
        num_additional_samples = self.num_frames_per_proc - new_num_frames_per_proc
        if num_additional_samples > 0:
            fill_exps_idx = np.random.randint(low=0,
                                              high=new_num_frames_per_proc,
                                              size=num_additional_samples)
            for enum,idx in enumerate(fill_exps_idx):
                self.obss[new_num_frames_per_proc+enum] = self.obss[idx]

            self.actions[new_num_frames_per_proc:] = self.actions[fill_exps_idx]
            self.rewards[new_num_frames_per_proc:] = self.rewards[fill_exps_idx]
            self.log_probs[new_num_frames_per_proc:] = self.log_probs[fill_exps_idx]
            self.values_ext[new_num_frames_per_proc:] = self.values_ext[fill_exps_idx]
            self.advantages_ext[new_num_frames_per_proc:] = self.advantages_ext[fill_exps_idx]
            self.returns_ext[new_num_frames_per_proc:] = self.returns_ext[fill_exps_idx]
            self.advantages_int[new_num_frames_per_proc:] = self.advantages_int[fill_exps_idx]
            self.advantages_total[new_num_frames_per_proc:] = self.advantages_total[fill_exps_idx]
            self.returns_int[new_num_frames_per_proc:] = self.returns_int[fill_exps_idx]

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        # exps.obs = [self.obss[i][j]
        #             for j in range(self.num_procs)
        #             for i in range(self.num_frames_per_proc)]
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        """
        # RECURRENCY
        if self.separated_actor_critic:
            if self.use_recurrence_actor:
                exps.memory_actor = self.memories_actor.transpose(0, 1).reshape(-1, *self.memories_actor.shape[2:])
                exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            if self.use_recurrence_critic:
                exps.memory_critic = self.memories_critic.transpose(0, 1).reshape(-1, *self.memories_critic.shape[2:])
                exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        else:
            if self.use_recurrence:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
                # T x P -> P x T -> (P * T) x 1
                exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        """
        
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # extrinsic stream (used normally)
        exps.value_ext = self.values_ext.transpose(0, 1).reshape(-1)
        exps.advantage_ext = self.advantages_ext.transpose(0, 1).reshape(-1)
        exps.returnn_ext = self.returns_ext.transpose(0, 1).reshape(-1)

        # additional intrinsic stream required when using two-streams instead of one
        exps.advantage_int = self.advantages_int.transpose(0, 1).reshape(-1)
        exps.advantage_total = self.advantages_total.transpose(0,1).reshape(-1)
        exps.returnn_int = self.returns_int.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        ########################################################################
        # LOGS
        ########################################################################
        weight_int_coef = self.int_coef
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": new_num_frames_per_proc,
            "return_int_per_episode": self.log_return_int[-keep:],
            "return_int_per_episode_norm": self.log_return_int_normalized[-keep:],
            "normalization_int_score": self.normalization_int_score,
            "episode_counter": self.episode_counter,
            "avg_return": mean(self.last_100return),
            "avg_return_int": mean(self.last_100return_int),
            "avg_steps": mean(self.last_100steps),
            "avg_success": mean(self.last_100success),
            "weight_int_coef": weight_int_coef,
            "predominance_ext_over_int": self.predominance_ext_over_int.mean().item(),
        }

        # update for new rollout collection
        self.log_done_counter = 0
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_return_int_normalized = self.log_return_int_normalized[-self.num_procs:]
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs, self.episode_info


    def evaluate_agent(self, num_episodes, reset_seed=False):
        """
            Evaluate the agent into a number of levels
        """
        average_success = np.zeros(num_episodes)
        average_steps = np.zeros(num_episodes)
        average_return = np.zeros(num_episodes)

        print('evaluation begins...')
        for e in range(num_episodes):
            
            next_seed = None
            if reset_seed:
                next_seed = np.random.randint(low=self.init_level_seed,
                                            high=self.init_level_seed + self.num_train_seeds) #used to determine next sampled level
            
            obs = self.ev_env.reset()
            rewards = []
            steps = 0

            while True:
                steps += 1

                preprocessed_obs = self.preprocess_obss(obs, device=self.device)

                if self.separated_actor_critic:
                    distribution, logits, _embedding = self.acmodel[0](preprocessed_obs)
                    value = self.acmodel[1](preprocessed_obs)
                else:
                    distribution, logits, value = self.acmodel(obs=preprocessed_obs)

                action = distribution.sample()                
                obs, reward, done, info = self.ev_env.step(actions=action,seeds=[next_seed])
                    
                reward = np.array(reward)
                rewards.append(reward)

                if np.array(done):
                    rewards = np.array(rewards).squeeze(1)
                    average_success[e] = 1 if np.sum(rewards) > 0 else 0
                    average_steps[e] = steps
                    average_return[e] = np.sum(rewards)
                    break
                
        print('evaluation finishes!')
        return np.mean(average_success), np.mean(average_steps),np.mean(average_return)

    @abstractmethod
    def update_parameters_ppo(self):
        pass

    @abstractmethod
    def update_parameters_imitation(self):
        pass
