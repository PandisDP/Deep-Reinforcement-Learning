
import random
import torch
import math
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
        return random.sample(self.memory,batch_size) 

    def can_provide_sample(self,batch_size):
        return len(self.memory)> batch_size               