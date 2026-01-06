'''
Created on 08/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
import abc

ENV_SINGLE_STATE = "***SINGLE_STATE***"

class Environment(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):

        ##初始化方法，设置一些属性，如学习者列表和步数等
        self._learners = {}
        self._episodes = 0
        self._steps = 0
        self._has_episode_ended = False
        self.replay_buffers = {}

        
    #return the actions available in the specified state
    #warning: if no action is available in a given state, then return [None]
    #选择指定状态下的动作，如果没有则返回0
    @abc.abstractmethod
    def get_state_actions(self, state):
        return
    
    #run an entire episode
    #运行整个
    @abc.abstractmethod
    def run_episode(self, max_steps=-1):
        pass

    #运行单个
    #run a single step within an episode
    @abc.abstractmethod
    def run_step(self):
        pass
    
    #check if the episode has ended
    #检查该轮是否运行结束
    @abc.abstractmethod
    def has_episode_ended(self):
        return
    
    #reset all attributes
    #重置所有属性


    def reset_all(self):
        self._episodes = 0
        
        for l in self._learners.values():
            l.reset_all()
        
    #reset all episode-related attributes
    def reset_episode(self):
        self._steps = 0
        self._has_episode_ended = False
        
        for l in self._learners.values():
            l.reset_episodic(self._episodes)
    
    #register a learner within the environment
    #在环境中注册智能体
    def register_learner(self, learner):
        if learner.get_name() in self._learners:
            raise Exception("Learner %s has already registered to %s!" % (learner.get_name(), self.__str__()))
        self._learners[learner.get_name()] = learner
    
    #efines what must happen when one prints the learnerd
    #定义返回智能体学习了什么
    def __str__(self):
        return "%s %s" % (self.__class__.__name__, self.__class__.__bases__[0].__name__)

        
