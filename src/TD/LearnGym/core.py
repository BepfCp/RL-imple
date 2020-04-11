from random import random, choice
import gym
from gym import Env
import numpy as np
from collections import namedtuple
from typing import List
import random
from tqdm import tqdm


class Transition(object):
    def __init__(self, s0, a0, reward:float, is_done:bool, s1):
        self.data = [s0, a0, reward, is_done, s1]

    @property
    def s0(self): return self.data[0]

    @property
    def a0(self): return self.data[1]

    @property
    def reward(self): return self.data[2]
    
    @property
    def is_done(self): return self.data[3]

    @property
    def s1(self): return self.data[4]

class Episode(object):
    def __init__(self, e_id=0):
        self.total_reward = 0
        self.trans_list = []
    
    @property
    def len(self):
        return len(self.trans_list)

    def push(self, trans:Transition):
        self.trans_list.append(trans)
        self.total_reward += trans.reward  # 不计衰减的总奖励
        return self.total_reward

    def is_complete(self):
        if self.len == 0: 
            return False 
        return self.trans_list[self.len-1].is_done

    def sample(self, batch_size = 1):   
        '''随即产生一个trans
        '''
        return random.sample(self.trans_list, k = batch_size)

class Experience(object):
    def __init__(self, capacity:int = 20000):
        self.capacity = capacity    # 容量：指的是trans总数量
        self.episodes = []          # episode列表
        self.next_id = 0            # 下一个episode的Id
        self.total_trans = 0        # 总的状态转换数量

    @property
    def len(self):
        return len(self.episodes)
        
    def push(self,trans):
        """push a trans
        """
        def _remove(self, index = 0):      
            '''remove an episode, defautly the first one.
            args: 
               the index of the episode to remove
            return:
               if exists return the episode else return None
            '''
            if index > self.len - 1:
                raise(Exception("invalid index"))
            if self.len > 0:
                episode = self.episodes[index]
                self.episodes.remove(episode)
                self.total_trans -= episode.len

        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity:
            episode = _remove()
        
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len-1]
        self.total_trans += 1
        return cur_episode.push(trans)      #return  total reward of an episode

    def sample(self, batch_size=1): # sample transition
        '''randomly sample some transitions from agent's experience.abs
        args:
            number of transitions need to be sampled
        return:
            list of Transition.
        '''
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num = 1):  # sample episode
        '''随机获取一定数量完整的Episode
        '''
        return random.sample(self.episodes, k = episode_num)

    @property
    def last_episode(self):
        if self.len > 0:
            return self.episodes[self.len-1]
        return None
    
class Agent(object):
    '''Base Class of Agent
    '''
    def __init__(self, env: Env = None, 
                       capacity = 10000):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env # 建立对环境对象的引用
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.S = [i for i in range(self.obs_space.n)]
        self.A = [i for i in range(self.action_space.n)]
        self.experience = Experience(capacity = capacity)
        # 有一个变量记录agent当前的state相对来说还是比较方便的。要注意对该变量的维护、更新
        self.state = None   # 个体的当前状态
    
    def policy(self, A, s = None, Q = None, epsilon = None):  #在这里应用学习算法，确定policy
        '''均一随机策略
        '''
        return random.sample(self.A, k=1)[0]
    
    def perform_policy(self, s, Q = None, epsilon = 0.05):
        action = self.policy(self.A, s, Q, epsilon)
        return int(action)
    
    def act(self, a0):
        s0 = self.state
        s1, r1, is_done, info = self.env.step(a0)
        trans = Transition(s0, a0, r1, is_done, s1)
        total_reward = self.experience.push(trans)
        self.state = s1
        return s1, r1, is_done, info, total_reward

    def learner(self,lambda_ = 0.9, gamma = 0.9, alpha = 0.5, epsilon = 0.2, display = False):
        '''这是一个没有学习能力的学习方法
        具体针对某算法的学习方法，返回值需是一个二维元组：(一个状态序列的时间步、该状态序列的总奖励)
        '''
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon = epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward  
                        
                        
    def learning(self, lambda_ = 0.9, epsilon = None, decaying_epsilon = True, gamma = 0.9, 
                 alpha = 0.1, max_episode_num = 800, display = False):
        total_time,  episode_reward, num_episode = 0,0,0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)):
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon:
                epsilon = 1.0 / (1 + num_episode)
            time_in_episode, episode_reward = self.learner(lambda_ = lambda_, \
                  gamma = gamma, alpha = alpha, epsilon = epsilon, display = display)
            total_time += time_in_episode
            num_episode += 1
            total_times.append(total_time)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
        return  total_times, episode_rewards, num_episodes
  