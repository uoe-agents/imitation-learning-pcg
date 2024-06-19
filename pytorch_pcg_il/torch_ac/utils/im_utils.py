import numpy as np

class GAE_Global(object):
    def __init__(self):

        # visitation counts
        self.counts = dict()
         # stores all observations of each episode
        self.episodes = dict()  # shape [num_samples, obs_dims*]
        # stores each episode's bonus --> {id_ep:bonus}
        self.episode_bonus = dict()
        # monitores how many episodes have been stored in the whole training
        self.episode_index = -1

    def add(self, obs):
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
        self.episode_index += 1
        self.episodes[self.episode_index] = obs

        # after visitation counts updated, stores the same score for all the samples of an episode
        self.update_bonus()

        return self.episode_index

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


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = batch_count + self.count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (tot_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
