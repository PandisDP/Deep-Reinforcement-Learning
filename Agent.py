
import random
import torch
import math
import numpy as np
#This class is used to manage the agent
class Agent():
    '''
    This class is used to manage the agent
    '''
    def __init__(self,strategy,num_actions,device):
        '''
        Params:
        strategy: The strategy used to select the actions
        num_actions: The number of actions that the agent can take
        device: The device used to run the agent'''
        self.step=0
        self.strategy=strategy
        self.num_actions= num_actions
        self.device=device

    def select_action(self,state,policy_net):
        '''
        This method is used to select an action for the agent
        throught exploration or exploitation
        Params:
        state: The state of the agent
        policy_net: The neural network used to calculate the Q-values
        Returns: The action selected by the agent'''
        rate= self.strategy.get_exploration_rate(self.step)
        self.step+=1
        if random.random()<rate:
            action= random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) #action
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)

class EpsilonGreedyStrategy():
    '''
    This class is used to manage the exploration rate of the agent
    '''
    def __init__(self,start,end,decay):
        '''
        Params:
        start: The initial exploration rate
        end: The final exploration rate
        decay: The decay of the exploration rate'''
        self.start= start
        self.end= end
        self.decay= decay

    def get_exploration_rate(self,step):
        '''
        This method is used to get the exploration rate of the agent
        Params:
        step: The step of the training process
        Returns: The exploration rate of the agent'''
        return self.end + (self.start - self.end)*math.exp(-step*self.decay)    

class ReplayMemory():
    '''
    This class is used to store the experiences of the agent
    '''
    def __init__(self,capacity):
        '''
        Params:
        capacity: The maximum number of experiences that the memory can store
        memory: List with the experiences
        count: The number of experiences stored in the memory'''
        self.capacity= capacity
        self.memory= []
        self.count=0
    def push(self,exp):
        '''
        This method is used to store an experience in the memory
        Params:
        exp: The experience to store in the memory
        '''
        if len(self.memory)< self.capacity:
            self.memory.append(exp)
        else:
            self.memory[self.count%self.capacity]=exp
        self.count+=1      
    def sample(self,batch_size):
        '''
        This method is used to get a sample of experiences from the memory
        Params:
        batch_size: The number of experiences to get from the memory
        Returns: A sample of experiences from the memory
        '''
        return random.sample(self.memory,batch_size),0,0

    def can_provide_sample(self,batch_size):
        '''
        This method is used to check if the memory has enough experiences to provide a sample
        Params:
        batch_size: The number of experiences to get from the memory
        Returns: True if the memory has enough experiences to provide a sample, False otherwise
        '''
        return len(self.memory)> batch_size 

class PrioritizedReplayMemory():
    '''
    This class is used to store the experiences of the agent with priorities
    '''
    def __init__(self, capacity, alpha=0.6):
        '''
        Params:
        capacity: The maximum number of experiences that the memory can store
        alpha: The exponent used to calculate the priority of the experiences
        beta: The exponent used to calculate the importance sampling weights
        beta_increment_per_sampling: The increment of beta for each sampling
        epsilon: A small value to avoid division by zero
        '''
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 1e-6

    def _get_priority(self, error):
        '''
        This method is used to calculate the priority of an experience
        Params:
        error: The error of the experience
        Returns: The priority of the experience'''
        return (error + self.epsilon) ** self.alpha

    def push(self, error, sample):
        '''
        This method is used to store an experience in the memory
        Params:
        error: The error of the experience
        sample: The experience to store in the memory
        '''
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        '''
        This method is used to get a sample of experiences from the memory
        Params:
        batch_size: The number of experiences to get from the memory
        Returns: A sample of experiences from the memory'''
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = min(max(s, 0), self.tree.total())
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        sampling_probabilities+=self.epsilon
        is_weights = np.power(self.tree.total() * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def update(self, idx, error):
        '''
        This method is used to update the priority of an experience
        Params:
        idx: The index of the experience
        error: The error of the experience
        '''
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def can_provide_sample(self, batch_size):
        '''
        This method is used to check if the memory has enough experiences to provide a sample
        Params:
        batch_size: The number of experiences to get from the memory
        Returns: True if the memory has enough experiences to provide a sample, False otherwise'''
        return self.tree.write >= batch_size  


class SumTree:
    '''
    This class is used to store the priorities of the experiences'''
    def __init__(self, capacity):
        '''
        Params:
        capacity: The maximum number of experiences that the memory can store
        tree: The binary tree used to store the priorities
        data: The experiences stored in the memory
        write: The number of experiences stored in the memory
        '''
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        '''
        This method is used to update the priorities of the experiences
        Params:
        idx: The index of the experience
        change: The change in the priority of the experience'''
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        '''
        This method is used to get the index of an experience
        Params:
        idx: The index of the experience
        s: The priority of the experience
        Returns: The index of the experience'''
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left] and self.tree[left] > 0:
            return self._retrieve(left, s)
        else:
            if self.tree[right] > 0:
                return self._retrieve(right, s - self.tree[left])
            else:
                return idx

    def total(self):
        '''
        This method is used to get the total priority of the experiences
        Returns: The total priority of the experiences'''
        return self.tree[0]

    def add(self, p, data):
        '''
        This method is used to store an experience in the memory
        Params:
        p: The priority of the experience
        data: The experience to store in the memory
        '''
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        '''
        This method is used to update the priority of an experience and 
        propagate the change to the parent nodes
        Params:
        idx: The index of the experience
        p: The priority of the experience
        '''
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        '''
        This method is used to get an experience from the memory
        Params:
        s: The priority of the experience
        Returns: The index, priority, and experience of the experience
        '''
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]                