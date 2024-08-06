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
    '''
    This class is used to manage the training process of the agent
    '''
    def __init__(self,device,enviroment,agent,memory_strategy,features):
        '''
        Params:
        device: The device used to run the agent
        enviroment: The enviroment of the game
        agent: The agent of the game
        memory_strategy: The memory strategy used to store the experiences
        features: The number of dimensiones of the game'''
        self.device= device
        self.env= enviroment
        self.memory= memory_strategy
        self.features= features
        self.agent= agent
        self.policy_net= DQN(self.features,self.env.get_number_of_actions()).to(self.device)
        self.target_net= DQN(self.features,self.env.get_number_of_actions()).to(self.device)
        self.qvalue= QValues(device)

    def training_priorized_memory(self,batch_size,gamma,target_update,
                                learning_rate,num_episodes,checkpoint_file='checkpoint.pth'):
        '''
        This method is used to train the agent using the Prioritized Memory
        Params:
        batch_size: The number of experiences to get from the memory
        gamma: The discount factor
        target_update: The number of episodes to update the target network
        learning_rate: The learning rate of the training process
        num_episodes: The number of episodes to train the agent
        checkpoint_file: The name of the file to save the checkpoint
        '''
        print('Training Process with Prioritized Memory')
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.target_net.eval()
        optimizer= th.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        episode_durations=[]
        episode_losses=[]
        total_timesteps = 0
        # Load checkpoint if exists and get the start episode
        start_episode=0
        try:
            start_episode= self.load_checkpoint(optimizer,checkpoint_file)
        except FileNotFoundError:
            print('No checkpoint found starting from scratch')    
        for episode in range(start_episode,num_episodes):
            self.env.reset()
            for timestep in count():
                state= self.env.get_state()
                action= self.agent.select_action(state,self.policy_net)
                reward,done= self.env.make_action(action)
                next_state= self.env.get_state()
                done = th.tensor([done], device=self.device, dtype=th.bool)
                with th.no_grad():
                    current_q_value = self.qvalue.get_current_i(self.policy_net, state, action)
                    next_q_value = self.qvalue.get_next_i(self.target_net, next_state)
                    target_q_value = reward + (gamma * next_q_value * (1 - done.float()))
                    error = abs(current_q_value - target_q_value).item()    
                self.memory.push(error, Experience(state, action, next_state, reward, done))
                if self.memory.can_provide_sample(batch_size):
                    experiences,idxs,is_weights= self.memory.sample(batch_size)
                    states,actions,rewards,next_states,is_done= self.__extract_tensors(experiences)
                    current_q_values= self.qvalue.get_current(self.policy_net,states,actions)
                    with th.no_grad():
                        next_q_values= self.qvalue.get_next(self.target_net,next_states,is_done)
                    target_q_values= (next_q_values*gamma)+rewards
                    is_weights= th.tensor(is_weights,dtype=th.float).unsqueeze(1).to(self.device)
                    loss = (is_weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1)
                                                    , reduction='none')).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_timesteps += 1
                    errors = th.abs(current_q_values - target_q_values.unsqueeze(1)).detach()
                    for idx, error in zip(idxs, errors):
                        self.memory.update(idx, error.item())    
                if done:
                    episode_durations.append(timestep)
                    if 'loss' in locals():
                        avg_loss = loss.item()
                    else:
                        avg_loss = 0   
                    print("Episode: ",episode," Average_Losses: ",avg_loss,
                        " Duration: ",timestep)  
                    self.save_checkpoint(episode,optimizer) 
                    break
                if total_timesteps % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())     

    def training_replay_memory(self,batch_size,gamma,target_update,learning_rate,
                            num_episodes,checkpoint_file='checkpoint.pth'):
        '''
        This method is used to train the agent using the Replay Memory
        Params:
        batch_size: The number of experiences to get from the memory
        gamma: The discount factor
        target_update: The number of episodes to update the target network
        learning_rate: The learning rate of the training process
        num_episodes: The number of episodes to train the agent
        checkpoint_file: The name of the file to save the checkpoint
        '''
        print('Training Process with Replay Memory')
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        optimizer= th.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        episode_durations=[]
        episode_losses=[]
        total_timesteps = 0
        # Load checkpoint if exists and get the start episode
        start_episode=0
        try:
            start_episode= self.load_checkpoint(optimizer,checkpoint_file)
        except FileNotFoundError:
            print('No checkpoint found starting from scratch')  
        for episode in range(start_episode,num_episodes):
            self.env.reset()
            for timestep in count():
                state= self.env.get_state()
                action= self.agent.select_action(state,self.policy_net)
                reward,done= self.env.make_action(action)
                next_state= self.env.get_state()
                done = th.tensor([done], device=self.device, dtype=th.bool)
                self.memory.push(Experience(state,action,next_state,reward,done))
                if self.memory.can_provide_sample(batch_size):
                    experiences,*_= self.memory.sample(batch_size)
                    states,actions,rewards,next_states,is_done= self.__extract_tensors(experiences)
                    current_q_values= self.qvalue.get_current(self.policy_net,states,actions)
                    with th.no_grad():
                        next_q_values= self.qvalue.get_next(self.target_net,next_states,is_done)
                    target_q_values= (next_q_values*gamma)+rewards
                    loss= F.mse_loss(current_q_values,target_q_values.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_timesteps += 1
                if done:
                    episode_durations.append(timestep)
                    if 'loss' in locals():
                        avg_loss = loss.item()
                    else:
                        avg_loss = 0   
                    print("Episode: ",episode," Average_Losses: ",avg_loss,
                        " Duration: ",timestep) 
                    self.save_checkpoint(episode,optimizer)
                    break
                if total_timesteps % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, episode, optimizer, filename='checkpoint.pth'):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'target_net_state_dict': self.target_net.state_dict()
        }
        th.save(checkpoint, filename)
        print(f'Checkpoint saved at episode {episode}')

    def load_checkpoint(self, optimizer, filename='checkpoint.pth'):
        if th.cuda.is_available():
            checkpoint = th.load(filename)
        else:
            checkpoint = th.load(filename, map_location=th.device('cpu'))
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print(f'Checkpoint loaded from episode {start_episode}')
        return start_episode    

    def __extract_tensors(self,experiences):
        batch = Experience(*zip(*experiences))
        states = th.cat(batch.state).to(self.device)
        actions = th.cat(batch.action).to(self.device)
        rewards = th.cat(batch.reward).to(self.device)
        next_states = th.cat(batch.next_state).to(self.device)
        final_states = th.tensor(batch.is_done, dtype=th.bool).to(self.device)
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

