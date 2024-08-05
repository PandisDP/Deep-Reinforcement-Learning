
import random
import torch
import math
import numpy as np
#This class is used to manage the agent
class Agent():
    def __init__(self,strategy,num_actions,device):
        self.step=0
        self.strategy=strategy
        self.num_actions= num_actions
        self.device=device

    def select_action(self,state,policy_net):
        rate= self.strategy.get_exploration_rate(self.step)
        self.step+=1
        if random.random()<rate:
            action= random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) #action
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)

class EpsilonGreedyStrategy():
    def __init__(self,start,end,decay):
        self.start= start
        self.end= end
        self.decay= decay

    def get_exploration_rate(self,step):
        return self.end + (self.start - self.end)*math.exp(-step*self.decay)    

class ReplayMemory():
    def __init__(self,capacity):
        self.capacity= capacity
        self.memory= []
        self.count=0
    def push(self,exp):
        if len(self.memory)< self.capacity:
            self.memory.append(exp)
        else:
            self.memory[self.count%self.capacity]=exp
        self.count+=1      
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size),0,0

    def can_provide_sample(self,batch_size):
        return len(self.memory)> batch_size 

class PrioritizedReplayMemory():
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 1e-6

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def push(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
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
            #print('bath',idx,p,self.tree.total(),data,a,b,s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        sampling_probabilities+=self.epsilon
        is_weights = np.power(self.tree.total() * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        #print('bath',idx,p,data)
        return batch, idxs, is_weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def can_provide_sample(self, batch_size):
        return self.tree.write >= batch_size  


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            if self.tree[right] > 0:
                return self._retrieve(right, s - self.tree[left])
            else:
                return idx

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]                