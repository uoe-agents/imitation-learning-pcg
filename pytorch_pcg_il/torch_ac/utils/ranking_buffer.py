import numpy as np
from gym.spaces import Discrete, Box

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class RankingBuffer(object):
    def __init__(self, ob_space, ac_space,
                 buffer_size, score_type,
                 w0, w1, w2,
                 rank_type='rapid',
                 score_function_type='linear_sum',
                 ):
        '''
        Args:
            w0: Weight for extrinsic rewards
            w1: Weight for local bonus
            w2: Weight for global bonus (sums of count-based exploration)
        '''

        # determine space dim
        self.ob_shape = ob_space
        self.ob_dim = 1
        print(self.ob_shape)
        for dim in self.ob_shape:
            self.ob_dim *= dim
        # self.ob_dim =ob_dim

        # determine action dim
        if isinstance(ac_space, Discrete):
            self.action_type = 'discrete'
        elif isinstance(ac_space, Box):
            self.action_type = 'box'
            self.ac_dim = ac_space.shape[0]
        else:
            ValueError('The action space is not supported.')

        self.size = buffer_size # max number of elements to be stored simultaneously
        self.data = None # buffer itself
        self.index = 0 # used to monitor number of elements in buffer
        self.score = 0
        self.score_type = score_type # 'discrete' or 'continious'
        self.score_function_type = score_function_type # 'linear_sum' or 'ext_dependant'
        self.ranking_type = rank_type

        # weights related to score calculation
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.wlongevity = 1e-10
        print('Buffer weights:', self.w0, self.w1, self.w2)
        print('Observation dimensions:',self.ob_dim)
        print('Action space:',self.action_type)


        # set format of data for access it latter
        self.setDataFormatIdx()

    def saveDemonstrationBuffer(self,path=[]):
        # store the buffer itself
        np.save(file=path, arr=self.data)
        # store the state-action tuples (to calculate W2 in the future)

    def loadDemonstrationBuffer(self,path=[]):
        data_buffer = np.load(file=path)
        self.data = data_buffer
        self.index = self.data.shape[0]

    def setDataFormatIdx(self):
        """
            Format of data:
            - observation
            - action
            - next_observation
            - reward (at that step)
            - dones
            - episode index
            - MC return of whole episode (extrinsic bonus)
            - local bonus
            - global bonus
            - total score
            - longevity (num episodes that has been in the buffer)

            We specify the idx in which those data can be accessed
        """
        self.idx_obs = 0
        self.idx_acs = self.idx_obs + self.ob_dim
        self.idx_rew = self.idx_acs + 1
        self.idx_nobs = self.idx_rew + 1
        self.idx_dones = self.idx_nobs + self.ob_dim
        self.idx_episode = self.idx_dones + 1
        self.idx_ret = self.idx_episode + 1
        self.idx_dret = self.idx_ret + 1
        self.idx_step_order = self.idx_dret + 1
        self.idx_local = self.idx_step_order + 1
        self.idx_global = self.idx_local + 1
        self.idx_totalscore = self.idx_global + 1
        self.idx_level = self.idx_totalscore + 1
        self.idx_regret = self.idx_level + 1
        self.idx_longevity = self.idx_regret + 1


    def prepareData(self,num_steps, obs, acs, rew, nobs, dones, ret, local_bonus, level, ep_idx):
        # eval_data = np.zeros((num_steps,303))
        # eval_data[:,self.idx_obs:self.idx_acs] = obs.astype(float).reshape(num_steps,-1)
        # eval_data[:,self.idx_acs:self.idx_rew] = acs
        # eval_data[:,self.idx_rew:self.idx_nobs] = np.expand_dims(rew, axis=1)
        # eval_data[:,self.idx_nobs:self.idx_dones] = nobs.astype(float).reshape(num_steps,-1)
        # eval_data[:,self.idx_dones:self.idx_episode] = np.expand_dims(dones, axis=1)
        # eval_data[:,self.idx_episode:self.idx_ret] = np.zeros((num_steps,1))
        # eval_data[:,self.idx_ret:self.idx_local] = np.expand_dims(np.repeat(ret,num_steps), axis=1)
        # eval_data[:,self.idx_local:self.idx_global] = np.expand_dims(np.repeat(local_bonus,num_steps), axis=1)
        # eval_data[:,self.idx_global:self.idx_totalscore] = np.expand_dims(np.repeat(0,num_steps), axis=1)
        # eval_data[:,self.idx_totalscore:self.longevity] = np.zeros((num_steps,1))
        # eval_data[:,self.longevity:] = np.zeros((num_steps,1))

        disc_ret = discount_with_dones(rewards=rew,dones=dones,gamma=0.99)

        data = np.concatenate((
            obs.astype(float).reshape(num_steps,-1),
            acs,
            np.expand_dims(rew, axis=1),
            nobs.astype(float).reshape(num_steps,-1),
            np.expand_dims(dones, axis=1),
            np.expand_dims(np.repeat(ep_idx,num_steps), axis=1),
            np.expand_dims(np.repeat(ret,num_steps), axis=1),
            np.expand_dims(disc_ret, axis=1),
            np.expand_dims(np.arange(num_steps), axis=1),
            np.expand_dims(np.repeat(local_bonus,num_steps), axis=1),
            np.expand_dims(np.repeat(0,num_steps), axis=1),
            np.zeros((num_steps,1)),
            np.expand_dims(np.repeat(level,num_steps), axis=1),
            np.zeros((num_steps,1)),
            np.zeros((num_steps,1)),
            ), axis=1)

        return data


    def getUpdatedGlobalBonus(self,current_episodes=[],update_all=True):
        if self.w2 >0:
            # Updates just the new upcoming episodes w2 score
            if len(current_episodes) > 0 and update_all==False:
                for e in current_episodes:
                    # get the experiences of that episode
                    idx_ = np.argwhere(self.data[:,self.idx_episode] == e).squeeze()
                    # get updated global score of all the experiences of A SINGLE EPISODE
                    global_bonus = self.counter.get_bonus([e])
                    self.data[idx_,self.idx_global] = global_bonus
            else:
                # get current episode index
                episode_idx = self.data[:,self.idx_episode].astype(int)
                # get updated global score of all the experiences stored in the buffer
                global_bonus = self.counter.get_bonus(episode_idx)
                # update all the experiences
                self.data[:,self.idx_global] = global_bonus

    def calulateEpisodeScore(self):
        rapid_score = 0
        if self.score_function_type == 'linear_sum':
            rapid_score = (self.w0 * self.data[:,self.idx_ret]) \
                        + (self.w1 * self.data[:,self.idx_local]) \
                        + (self.w2 * self.data[:,self.idx_global]) 

        # possible extensions with a different weighting function
        elif self.score_function_type == 'ext_dependant':
            rapid_score = (self.w0 * self.data[:,self.idx_ret]) \
                        + (self.w1 * self.data[:,self.idx_local]) \
                        + (self.w2 * self.data[:,self.idx_global])

        return rapid_score

    def sort_and_drop(self,counter_global=None,
                      current_episodes=[],current_levels=[],
                      online_return = {}, online_gae = {}):

        # 1. ***CALCULATE SCORES*** (def:all; or just new episodes)
        # 1.1.calculate global bonus
        if self.w2 > 0:
            # copy dict/nn of counter
            self.counter = counter_global
        self.getUpdatedGlobalBonus(current_episodes=current_episodes)

        # 1.2.calculate regret
        # self.getUpdatedGlobalRegret(current_episodes=current_episodes,
        #                             online_return=online_return,
        #                             online_gae=online_gae)

        # Update rapid_score with new global and regret bonuses
        rapid_score = self.calulateEpisodeScore()

        total_score = rapid_score - (self.wlongevity *self.data[:,self.idx_longevity])
        self.data[:,self.idx_totalscore] = total_score

        # 2. ***RANKING/DROP STRATEGY***
        if self.ranking_type == 'store_one_episode':
            # print('stored levels:',current_levels)
            list_exps_pop = []

            for l in current_levels:
                
                # get the idx of trajs of same levels
                _idx = np.argwhere(self.data[:,self.idx_level] == l).squeeze()
                # select the episode_idx we are interested in
                _episodes_idx = np.unique(self.data[_idx,self.idx_episode])
                # print('current level {} has the next episode idx {}'.format(l,_episodes_idx))

                episode_with_max_score = 0
                max_score = 0
                min_steps = 100000
                episodes_to_exps_dict = {}

                for e in _episodes_idx:
                    # store in dict
                    episodes_to_exps_dict[e]= _exps = np.argwhere(self.data[:,self.idx_episode] == e).squeeze()
                    _score = self.data[_exps[0],self.idx_ret]
                    _steps = len(_exps)
                    # print('epidx:{}, score:{}'.format(e,_score))

                    if (_score > max_score) or (_score==max_score and _steps<min_steps):
                        max_score = _score
                        min_steps = _steps
                        episode_with_max_score = e
                        # print('ep{} new max_score{} and min steps:{}'.format(e,max_score,min_steps))
                
                for k,v in episodes_to_exps_dict.items():
                    if k != episode_with_max_score:
                        # print('borrar episodes con id {}'.format(k))
                        list_exps_pop.extend(v)

            # print('data shape:',self.data.shape)
            self.data = np.delete(arr=self.data, obj=list_exps_pop,axis=0)
            # print('data shape post:',self.data.shape)
            # u,uc = np.unique(self.data[:,self.idx_level],return_counts=True)
            # print('unique levels {}, with coutns: {}'.format(u,uc))
            # print()

        elif self.ranking_type == 'rapid':
            # Generate sorted idx based on the score
            sort_idx = self.data[:,self.idx_totalscore].argsort()

            # Keep in buffer only at MAX self.size samples
            self.data = self.data[sort_idx][-self.size:]

        elif self.ranking_type == 'fifo':
            # Generate sorted idx based First Input First Output
            sort_idx = self.data[:,self.idx_longevity].argsort()
            # Keep in buffer only at MAX self.size samples
            self.data = self.data[sort_idx][-self.size:]

        # update number of elements stored in the buffer
        self.index = self.data.shape[0]
        # update the score (avg) of elements stored in buffer
        self.score = np.mean(rapid_score)

    def insert(self, obs, acs, rew, nobs, dones, ret, level, steps, episode_idx=0,
              mem=[]):
        # calculate local bonus
        if self.w1 > 0:
            local_bonus = get_local_bonus(obs, self.score_type)
        else:
            local_bonus = 0.0

        # expand dims to have the same as obs
        if self.action_type == 'discrete':
            _ac_data = np.expand_dims(acs, axis=1)
        elif self.action_type == 'box':
            _ac_data = acs

        # prepare data --> _data == new incoming experiences of a single episode
        num = obs.shape[0] # number of steps in episode
        _data = self.prepareData(num_steps=num, obs=obs, acs=_ac_data, rew=rew, nobs=nobs,
                                 dones=dones, ret=ret, local_bonus = local_bonus,
                                 level=level, ep_idx=episode_idx)

        # Fill buffer
        if (self.data is None):
            self.data = _data # if is empty
        else:
            self.data = np.concatenate((self.data, _data), axis=0) # if not empty, concatenate the new data
            self.data[:,self.idx_longevity] = self.data[:,self.idx_longevity] + 1 # update counter of episodes --> the more negative the value, the more time in buffer (used for sorting ease reasons)

        # update number of elements stored in the buffer
        self.index = self.data.shape[0]

    def getIdxEpisodesUsed(self):
        return np.unique(self.data[:,self.idx_episode])

    def sample_bc(self, batch_size, sample_full_traj=False):
        """
            Sample data for Behavioral Clonning loss:
            - Obs
            - Action
            - Discounted Return
        """
        if sample_full_traj==True:
            # get idx of trajectories
            different_traj = np.unique(self.data[:,self.idx_episode]) #idx_episodes takes also into account possible multiple trajectories of the same level
            num_sampled_exps = 0
            idx = []
            while num_sampled_exps < batch_size:
                # randomly select one trajectory
                idx_traj = np.random.choice(different_traj)
                # get idx of all exps of that trajectory
                idx_traj_exps = np.argwhere(self.data[:,self.idx_episode] == idx_traj).squeeze(axis=1)
                num_sampled_exps += len(idx_traj_exps)
                # those samples are not ordered in sequence --> order them!
                # originalsequenceorder = self.data[idx_traj_exps,self.idx_step_order]
                sort_idx = self.data[idx_traj_exps,self.idx_step_order].argsort() # get the idx sorted ascendant taking into account the step order attribute (the order we want)
                idx_traj_exps_sorted = idx_traj_exps[sort_idx]
                # print('IDX OF SAMPLES:',idx_traj_exps)
                # print('originalsequenceorder',originalsequenceorder)
                # print('sorted',idx_traj_exps[sort_idx])

                try:
                    # extend the idx to sample all those data later out-loop
                    idx.extend(idx_traj_exps_sorted)
                except TypeError:
                    print('Error again')
                    print('idx diff trajectory:',different_traj)
                    print('idx_traj:',idx_traj)
                    print('new exps to extend:',idx_traj_exps)
                    print('without squeeze:',np.argwhere(self.data[:,self.idx_episode] == idx_traj))

            # update batch size used in this sampling for further format processing
            batch_size = len(idx)
        else:
            idx = np.random.choice(range(0,self.index), batch_size)

        # sample the data
        sampled_data = self.data[idx]

        # 1.get observations
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)
        # obs = obs.reshape(batch_size,self.ob_dim)

        # 2.get actions
        if self.action_type == 'discrete':
            acs = sampled_data[:,self.ob_dim].astype(int)
        elif self.action_type == 'box':
            acs = sampled_data[:,self.ob_dim:self.ob_dim+self.ac_dim]

        # 3.get discounted return
        dret = sampled_data[:,self.idx_dret]

        # 4. get dones
        dones = sampled_data[:, self.idx_dones]

        # 5. get next_obs
        next_obs = sampled_data[:,self.idx_nobs:self.idx_nobs+self.ob_dim]
        next_obs = next_obs.reshape((batch_size,) + self.ob_shape)

        # get influence of each component on the sampled batch
        w0_weight = self.w0*sampled_data[:,self.idx_ret]
        w1_weight = self.w1*sampled_data[:,self.idx_local]
        w2_weight = self.w2*sampled_data[:,self.idx_global]

        influence_of_scores = {}
        influence_of_scores['w0'] = np.mean(w0_weight)
        influence_of_scores['w1'] = np.mean(w1_weight)
        influence_of_scores['w2'] = np.mean(w2_weight)

        # seeds dict --> gives % of samples used by each level in the current batch
        seeds_dict = {}
        seed_levels = sampled_data[:,self.idx_level]
        for k in np.unique(seed_levels):
            seeds_dict[k] = np.sum(seed_levels == int(k))/batch_size

        return idx, obs, acs, dret, dones, next_obs, influence_of_scores, seeds_dict

    def sample_full_experience(self, batch_size):
        """
            Sample data for Behavioral Clonning loss:
            - Obs
            - Action
            - Nobs
            - Reward
            - Dones
        """
        idx = np.random.choice(range(0,self.index), batch_size)
        # idx = [0,1,2,3,4]
        sampled_data = self.data[idx]

        # get observations
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)

        # get actions
        if self.action_type == 'discrete':
            acs = sampled_data[:,self.ob_dim].astype(int)
        elif self.action_type == 'box':
            acs = sampled_data[:,self.ob_dim:self.ob_dim+self.ac_dim]

        # get next_observations
        nobs = sampled_data[:, self.idx_nobs:self.idx_nobs+self.ob_dim]
        nobs = nobs.reshape((batch_size,) + self.ob_shape)

        # get rewards
        rews = sampled_data[:, self.idx_rew]

        # get dones
        dones = sampled_data[:, self.idx_dones]

        return idx, obs, acs, rews, nobs, dones

    def reset(self):
        self.data = None
        self.index = 0

    def getIndex(self):
        return self.index


def get_local_bonus(obs, score_type):
    if score_type == 'discrete':
        obs = obs.reshape((obs.shape[0], -1))
        unique_obs = np.unique(obs, axis=0)
        total = obs.shape[0]
        unique = unique_obs.shape[0]
        score = float(unique) / total
    elif score_type == 'continious':
        obs = obs.reshape((obs.shape[0], -1))
        obs_mean = np.mean(obs, axis=0)
        score =  np.mean(np.sqrt(np.sum((obs - obs_mean) * (obs -obs_mean), axis=1)))
    else:
        raise ValueError('Score type {} is not defined'.format(score_type))

    return score
