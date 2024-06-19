from multiprocessing import Process, Pipe
import gym
import numpy as np

def worker(conn, env, env_dict):
    """
        Not implemented (if you desire to parallelize, here is the place)
    """
    while True:
        # cmd, data, reset_seed = conn.recv()
        cmd, data = conn.recv()
        # print('data (AKA action):',data)

        env_key = list(env_dict.keys())[0]
        # single step
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                # reset
                obs = env.reset()
            conn.send((obs, reward, done, info))

        # reset
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, env_dict):
        assert len(envs) >= 1, "No environment given."
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        # print('\n\nenv_dict:',env_dict)

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, env_dict))
            p.daemon = True
            p.start()
            remote.close()
            
        # Initialize seeds for all environments
        self.current_seeds = [None] * len(envs)  # Default None, meaning random seed

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions, seeds=None):
        # update current seed value
        self.current_seeds = seeds
            
        # For Env1,Env2,Env3...
        for local, action in zip(self.locals, actions[1:], seeds[1:]):
            local.send(("step", action))

        # Env0
        obs, reward, done, info = self.envs[0].step(actions[0])

        # reset environment (with selected seed or random/automated)
        if done:
            # select next level/seed
            
            if self.current_seeds[0] is not None:
                self.envs[0].seed(int(self.current_seeds[0]))
            
            # reset as usual
            obs = self.envs[0].reset()

        # gather other parallel agents info
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])

        return results

    def render(self,mode):
        return self.envs[0].render(mode)
        # raise NotImplementedError
