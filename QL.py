
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class QValues():
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = th.device("mps" if th.backends.mps.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net,states,actions):
        value_=policy_net(states).gather(dim=1,index=actions.unsqueeze(-1))
        return value_
    
    @staticmethod
    def get_next(target_net,next_states,is_done):
        next_q_values= torch.zeros(len(next_states)).to(QValues.device)
        non_final_mask= ~is_done
        non_final_next_states= next_states[non_final_mask]
        if len(non_final_next_states)>0:
            with torch.no_grad():
                next_q_values[non_final_mask]= target_net(non_final_next_states).max(dim=1)[0]
        return next_q_values
        
class DQN(nn.Module):
    def __init__(self,feature_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_size,out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.out= nn.Linear(in_features=128 ,out_features=num_actions)

    def forward(self,t): 
    
        if t.dim()==1:
            t= t.unsqueeze(1)
        t=t.float()  
        t= F.relu(self.fc1(t))
        t= F.relu(self.fc2(t))
        t= F.relu(self.fc3(t))
        t= self.out(t)
        return t    
