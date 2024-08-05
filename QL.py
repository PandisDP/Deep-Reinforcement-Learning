
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class QValues():
    '''
    This class is used to manage the Q-values of the agent
    '''
    def __init__(self,device):
        '''
        Params:
        device: The device used to run the agent'''
        self.device= device
    def get_current(self,policy_net,states,actions):
        '''
        This method is used to get the Q-values of the current state
        Params:
        policy_net: The neural network used to calculate the Q-values
        states: The states of the agent
        actions: The actions of the agent
        Returns: The Q-values of the current state'''
        return policy_net(states).gather(dim=1,index=actions.unsqueeze(-1))
    
    def get_current_i(self,policy_net,state,action):
        '''
        This method is used to get the Q-values of the current state
        Params:
        policy_net: The neural network used to calculate the Q-values
        state: The state of the agent
        action: The action of the agent
        Returns: The Q-values of the current state'''
        return policy_net(state).gather(1, th.tensor([[action]], device=self.device))
    
    def get_next_i(self, target_net, next_state):
        '''
        This method is used to get the Q-values of the next state
        Params:
        target_net: The target neural network used to calculate the Q-values
        next_state: The next state of the agent
        Returns: The Q-values of the next state'''
        return target_net(next_state).max(1)[0].unsqueeze(1)
    
    def get_next(self,target_net,next_states,is_done):
        '''
        This method is used to get the Q-values of the next state
        Params:
        target_net: The target neural network used to calculate the Q-values
        next_states: The next states of the agent
        is_done: A boolean array that indicates if the episode is done
        Returns: The Q-values of the next state'''
        next_q_values= torch.zeros(len(next_states)).to(self.device)
        non_final_mask= ~is_done
        non_final_next_states= next_states[non_final_mask]
        if len(non_final_next_states)>0:
            next_q_values[non_final_mask]= target_net(non_final_next_states).max(dim=1)[0]
        return next_q_values
        
class DQN(nn.Module):
    '''
    This class is used to manage the neural network of the agent
    This arquiteture is used in the Deep Q-Learning algorithm but is possible
    change for diferentes problems or games.
    '''
    def __init__(self,feature_size, num_actions):
        '''
        Params:
        feature_size: The size of the features
        num_actions: The number of actions that the agent can take'''
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_size,out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.out= nn.Linear(in_features=128 ,out_features=num_actions)

    def forward(self,t): 
        '''
        This method calculates the Q-values of the agent.
        
        Params:
        t (Tensor): The input features of the agent.
        Returns:
        Tensor: The Q-values of the agent.
        '''
    
        if t.dim()==1:
            t= t.unsqueeze(1)
        t=t.float()  
        t= F.relu(self.fc1(t))
        t= F.relu(self.fc2(t))
        t= F.relu(self.fc3(t))
        t= self.out(t)
        return t    
