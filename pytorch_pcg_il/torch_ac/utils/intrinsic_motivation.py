#!/usr/bin/env python3

import math
import numpy as np
import torch.nn.functional as F

class CountModule():
    def __init__(self, state_action = False):
        self.state_action = state_action # define if depends also to the state
        self.counts = {}

    def visualize_counts(self):
        print('\nDict count:')
        for e,(k,v) in enumerate(self.counts.items()):
            print('Hash state:{} of len {} with Values:{}'.format(e,len(k),v))

    def check_ifnot_already_visited(self,next_obs,actions):
        """
            Used to generate determine if a state has been visited
            (with inverse logic)
        """
        tup = self.encode_state(next_obs,actions)
        if tup in self.counts:
            return 0 # if visited, mask=0
        else:
            return 1 # if not visited, mask=1

    def compute_intrinsic_reward(self,obs,next_obs,actions):
        """
            Generates the Intrinsic reward bonus based on the encoded state/tuple
            -Accepts a single observation
        """
        tup = self.encode_state(next_obs,actions)
        if tup in self.counts:
            return 1/math.sqrt(self.counts[tup])
        else:
            return 1

    def get_number_visits(self,obs,actions):
        tup = self.encode_state(obs,actions)
        return self.counts[tup]

    def update(self,obs,next_obs,actions):
        """
            Add samples to the bins;
                -It is prepared to catch inputs of shape [batch_size, -1]
                -i.e. [2048,7,7,3]
        """

        for idx,(o,a) in enumerate(zip(next_obs,actions)):
            tup = self.encode_state(o,a)
            if tup in self.counts:
                self.counts[tup] += 1
            else:
                self.counts[tup] = 1

    def encode_state(self,state,action):
        """
            Encodes the state in a tuple or taking also into account the action
        """
        state = state.view(-1).tolist()
        if self.state_action:
            return (tuple(state),action)
        else:
            return (tuple(state))

    def reset(self):
        """
            Re-init of counts
        """
        self.counts = {}

class Counter_Global(object):
    def __init__(self):

        # visitation counts
        self.counts = dict()
         # stores all observations of each episode
        self.episodes = dict()  # shape [num_samples, obs_dims*]
        # stores each episode's bonus --> {id_ep:bonus}
        self.episode_bonus = dict()
        # monitores how many episodes have been stored in the whole training
        # self.episode_index = -1

    def add(self, obs, episode_index):
        """
            Adds a visitation count for the input batch of observations
            -Updates the number of episodes
            -Saves in dictionary the observations
            -Updates the bonus of all the stores episodes in self.episodes
        """
        for ob in obs:
            ob = tuple(ob)
            if ob not in self.counts:
                self.counts[ob] = 1
            else:
                self.counts[ob] += 1
        # self.episode_index += 1
        self.episodes[episode_index] = obs

        # after visitation counts updated, stores the same score for all the samples of an episode
        self.update_bonus()

        # return self.episode_index

    def update_bonus(self):
        """
            Updates the episode bonus of all the stored experiences in self.episodes
        """
        for idx in self.episodes:
            bonus = []
            obs = self.episodes[idx]
            counter = 0
            # for each episode, update bonus
            for ob in obs:
                counter += 1
                ob = tuple(ob)
                count = self.counts[ob]
                bonus.append(count)
            bonus = 1.0 / np.sqrt(np.array(bonus))
            bonus = np.mean(bonus)
            self.episode_bonus[idx] = bonus

    def get_bonus(self, idxs):
        """
            Get bonus for all the experiences

            -Resize dictionaries of self.episodes and self.episode_bonus to only store info of episodes stored at the buffer
            # select only those episodes that are inside the idxs (that are taken from the buffer)
            # -Not all the episodes that have been visited in the train are stored! -- Not necessary, as the counts with which the
            bonus is calculated, is never resetted
        """
        # print('idx:',idxs)
        # print('unique:',np.unique(idxs))
        # print('episodes.keys()',self.episodes.keys())
        # print('episode_bonus.keys()',self.episode_bonus.keys())

        bonus = []
        for idx in idxs:
            bonus.append(self.episode_bonus[idx])
        return np.array(bonus)

    def updateEpisodeBuffers(self,idxs_being_used):
        self.episodes = {k:self.episodes[k] for k in idxs_being_used}
        self.episode_bonus = {k:self.episode_bonus[k] for k in idxs_being_used}

    def get_EpisodeBuffer(self):
        """
            returns the dict_states in each episode_idx
            -transform the observation/state from numpy to list
            --Why?(JSON does not serialize ndarrays)
        """
        return {k:v.tolist() for k,v in self.episodes.items()}

    def get_CountsDict(self):
        return {str(k):v for k,v in self.counts.items()}

    def set_CountsDict(self, dict):
        """
            https://www.geeksforgeeks.org/python-convert-a-string-representation-of-list-into-list/
            -convert str(list) to tuple so that the key can be imported correctly with ast
        """
        import ast

        # for k,v in dict.items():
        #     newk = ast.literal_eval(k)
            # print(type(k))
            # print(type(newk))
            # print(k)
            # print(newk)
            # self.counts[newk] = v
            # input()
        self.counts = {ast.literal_eval(k):v for k,v in dict.items()}

class BeBold():
    def __init__(self, state_action = False):
        self.state_action = state_action # define if depends also to the state
        self.counts = {}

    def visualize_counts(self):
        print('\nDict count:')
        for e,(k,v) in enumerate(self.counts.items()):
            print('Hash state:{} of len {} with Values:{}'.format(e,len(k),v))

    def check_ifnot_already_visited(self,next_obs,actions):
        """
            Used to generate determine if a state has been visited
            (with inverse logic)
        """
        tup = self.encode_state(next_obs,actions)
        if tup in self.counts:
            return 0 # if visited, mask=0
        else:
            return 1 # if not visited, mask=1

    def compute_intrinsic_reward(self,obs,next_obs,actions):
        """
            Generates the Intrinsic reward bonus based on the encoded state/tuple
            -Accepts a single observation
        """
        current_tup = self.encode_state(obs)
        next_tup = self.encode_state(next_obs)
        if next_tup in self.counts:
            if current_tup in self.counts:
                return max((1/self.counts[next_tup]) - (1/self.counts[current_tup]),0)
            else:
                return max((1/self.counts[next_tup]) - 1, 0)
        else:
            if current_tup in self.counts:
                return max(1 - (1/self.counts[current_tup]) ,0)
            else:
                return 1

    def get_number_visits(self,obs,actions):
        tup = self.encode_state(obs,actions)
        return self.counts[tup]

    def update(self,obs,next_obs,actions):
        """
            Add samples to the bins;
                -It is prepared to catch inputs of shape [batch_size, -1]
                -i.e. [2048,7,7,3]
        """

        for idx,(o,a) in enumerate(zip(next_obs,actions)):
            tup = self.encode_state(o,a)
            if tup in self.counts:
                self.counts[tup] += 1
            else:
                self.counts[tup] = 1

    def encode_state(self,state,action=None):
        """
            Encodes the state in a tuple or taking also into account the action
        """
        state = state.view(-1).tolist()
        if self.state_action:
            return (tuple(state),action)
        else:
            return (tuple(state))

    def reset(self):
        """
            Re-init of counts
        """
        self.counts = {}
