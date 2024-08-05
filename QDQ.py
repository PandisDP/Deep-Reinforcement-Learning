from QL import DQN,QValues
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from itertools import count
import torch.nn.functional as F
from collections import namedtuple
from IPython import display

Experience= namedtuple('Experience',('state','action','next_state','reward','is_done'))

class QDQ:
    def __init__(self,device,enviroment,agent,memory_strategy,features):
        self.device= device
        self.env= enviroment
        self.memory= memory_strategy
        self.features= features
        self.agent= agent
        self.policy_net= DQN(self.features,self.env.get_number_of_actions()).to(self.device)
        self.target_net= DQN(self.features,self.env.get_number_of_actions()).to(self.device)
        self.qvalue= QValues(device)

    def training_priorized_memory(self,batch_size,gamma,target_update,learning_rate,num_episodes):
        print('Training Process with Prioritized Memory')
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.target_net.eval()
        optimizer= th.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        episode_durations=[]
        episode_losses=[]
        total_timesteps = 0
        for episode in range(num_episodes):
            self.env.reset()
            episode_losses=[]
            total_loss= 0
            loss_count= 0
            for timestep in count():
                state= self.env.get_state()
                action= self.agent.select_action(state,self.policy_net)
                reward,done= self.env.make_action(action)
                next_state= self.env.get_state()
                # Calculate the error for the priority
                with th.no_grad():
                    current_q_value = self.policy_net(state).gather(1, th.tensor([[action]]))
                    next_q_value = self.target_net(next_state).max(1)[0].unsqueeze(1)
                    target_q_value = reward + (gamma * next_q_value * (1 - done))
                    error = abs(current_q_value - target_q_value).item()
                self.memory.push(error, Experience(state, action, next_state, reward, done))
                #memory.push(Experience(state,action,next_state,reward,done))
                if self.memory.can_provide_sample(batch_size):
                    experiences,idxs,is_weights= self.memory.sample(batch_size)
                    states,actions,rewards,next_states,is_done= self.__extract_tensors(experiences)
                    current_q_values= self.qvalue.get_current(self.policy_net,states,actions)
                    with th.no_grad():
                        next_q_values= self.qvalue.get_next(self.target_net,next_states,is_done)
                    target_q_values= (next_q_values*gamma)+rewards
                    is_weights= th.tensor(is_weights,dtype=th.float).unsqueeze(1)
                    #loss= F.mse_loss(current_q_values,target_q_values.unsqueeze(1))
                    loss = (is_weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1)
                                                    , reduction='none')).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Update priorities
                    errors = th.abs(current_q_values - target_q_values.unsqueeze(1)).detach().numpy()
                    for idx, error in zip(idxs, errors):
                        self.memory.update(idx, error)
                    episode_losses.append(loss.item())
                    total_timesteps += 1
                if done:
                    episode_durations.append(timestep)
                    print("Episode: ",episode," Average_Losses: ",np.mean(episode_losses),
                        " Duration: ",timestep)   
                    break
                if total_timesteps % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())     

    def training_replay_memory(self,batch_size,gamma,target_update,learning_rate,num_episodes):
        print('Training Process with Replay Memory')
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        optimizer= th.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        episode_durations=[]
        episode_losses=[]
        total_timesteps = 0
        for episode in range(num_episodes):
            self.env.reset()
            episode_losses=[]
            total_loss= 0
            loss_count= 0
            for timestep in count():
                state= self.env.get_state()
                action= self.agent.select_action(state,self.policy_net)
                reward,done= self.env.make_action(action)
                next_state= self.env.get_state()
                self.memory.push(Experience(state,action,next_state,reward,done))
                if self.memory.can_provide_sample(batch_size):
                    experiences,*_= self.memory.sample(batch_size)
                    #print(experiences)
                    states,actions,rewards,next_states,is_done= self.__extract_tensors(experiences)
                    current_q_values= self.qvalue.get_current(self.policy_net,states,actions)
                    with th.no_grad():
                        next_q_values= self.qvalue.get_next(self.target_net,next_states,is_done)
                    target_q_values= (next_q_values*gamma)+rewards
                    loss= F.mse_loss(current_q_values,target_q_values.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    episode_losses.append(loss.item())
                    total_timesteps += 1
                if done:
                    episode_durations.append(timestep)
                    print("Episode: ",episode," Average_Losses: ",np.mean(episode_losses),
                        " Duration: ",timestep)
                    break
                if total_timesteps % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())


    def __extract_tensors(self,experiences):
        batch = Experience(*zip(*experiences))
        states = th.cat(batch.state)
        actions = th.cat(batch.action)
        rewards = th.cat(batch.reward)
        next_states = th.cat(batch.next_state)
        final_states = th.tensor(batch.is_done, dtype=th.bool)
        return states, actions, rewards, next_states, final_states

    def __get_moving_avg(self,values,period):
        values = th.tensor(values,dtype=th.float)
        if len(values)>=period:
            moving_avg= values.unfold(dimension=0,size=period,step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg= th.cat((th.zeros(period-1),moving_avg))
            return moving_avg
        else:
            moving_avg= th.zeros(len(values))
            return moving_avg
        
    def __plot(self,values,moving_avg_period):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(values)
        moving_avg= self.get_moving_avg(values,moving_avg_period)
        plt.plot(moving_avg)
        plt.pause(0.001)
        #print("Episode", len(values),"\n",moving_avg_period,"episode moving avg:", moving_avg[-1])
        display.clear_output(wait=True)            

