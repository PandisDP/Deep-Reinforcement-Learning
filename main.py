from Base0 import wSum , NeuralNetwork
from  Games import Field
from QL import DQN,QValues
from Agent import Agent ,EpsilonGreedyStrategy,ReplayMemory
from Environment import EnvManager
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from itertools import count
import torch.nn.functional as F
from collections import namedtuple
from IPython import display
import torch


Experience= namedtuple('Experience',('state','action','next_state','reward','is_done'))

def task6():
    size=10
    start_position=(9,0) # (9,0)
    item_pickup=(1,1)# (1,1)
    item_dropoff=(8,8) # (8,8)
    zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    #zones_block=[]
    batch_size= 128
    features=1
    gamma= 0.99
    eps_start= 1
    eps_end= 0.01
    eps_decay= 0.001
    target_update= 5000
    memory_size= 20000
    lr= 0.001
    num_episodes= 10000

    # Usar GPU si está disponible
    #device = th.device("mps" if th.backends.mps.is_available() else "cpu")
    device= th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    em= Field(device,size,start_position,item_pickup,item_dropoff,zones_block,'Episodes')
    strategy= EpsilonGreedyStrategy(eps_start,eps_end,eps_decay)
    agent= Agent(strategy,em.get_number_of_actions(),device)
    memory= ReplayMemory(memory_size)
    policy_net= DQN(features,em.get_number_of_actions()).to(device)
    target_net= DQN(features,em.get_number_of_actions()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer= th.optim.Adam(policy_net.parameters(),lr=lr)
    episode_durations=[]
    episode_losses=[]
    total_timesteps = 0
    for episode in range(num_episodes):
        em.reset()
        episode_losses=[]
        total_loss= 0
        loss_count= 0
        for timestep in count():
            state= em.get_state()
            action= agent.select_action(state,policy_net)
            reward,done= em.make_action(action)
            next_state= em.get_state()
            memory.push(Experience(state,action,next_state,reward,done))
            if memory.can_provide_sample(batch_size):
                experiences= memory.sample(batch_size)
                states,actions,rewards,next_states,is_done= extract_tensors(experiences)
                current_q_values= QValues.get_current(policy_net,states,actions)
                with torch.no_grad():
                    next_q_values= QValues.get_next(target_net,next_states,is_done)
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
                target_net.load_state_dict(policy_net.state_dict())        
                
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)
    final_states = torch.tensor(batch.is_done, dtype=torch.bool)
    return states, actions, rewards, next_states, final_states

def get_moving_avg(values,period):
    values = th.tensor(values,dtype=th.float)
    if len(values)>=period:
        moving_avg= values.unfold(dimension=0,size=period,step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg= th.cat((th.zeros(period-1),moving_avg))
        return moving_avg
    else:
        moving_avg= th.zeros(len(values))
        return moving_avg
    
def plot(values,moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    moving_avg= get_moving_avg(values,moving_avg_period)
    plt.plot(moving_avg)
    plt.pause(0.001)
    #print("Episode", len(values),"\n",moving_avg_period,"episode moving avg:", moving_avg[-1])
    display.clear_output(wait=True)

def task4():
    device= th.device("cuda" if th.cuda.is_available() else "cpu")  
    em= EnvManager(device)
    em.reset()
    screen= em.render()
    plt.figure()
    plt.imshow(screen)
    plt.show()
    screen= em.get_processed_screen()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(),interpolation='none')
    plt.show()
    print(device)

def test():
# Verificar si MPS está disponible
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Ejemplo simple para verificar que MPS funciona
    x = torch.rand(5, 3).to(device)
    print(x)  

def task3():
    inputDim = 10
    n=1000
    X = np.random.rand(n,inputDim)
    y= np.random.randint(0,2,n)
    tensor_x= th.Tensor(X)
    tensor_y= th.Tensor(y)
    Xy= TensorDataset(tensor_x,tensor_y)
    Xy_loader= DataLoader(Xy,batch_size=16,shuffle=True,drop_last=True)   
    model = nn.Sequential(
        nn.Linear(inputDim,200),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=200),
        nn.Dropout(0.5),
        nn.Linear(200,100),
        nn.Tanh(),
        nn.BatchNorm1d(num_features=100),
        nn.Linear(100,1),
        nn.Sigmoid()
    ) 
    optimizer= th.optim.Adam(model.parameters(),lr=0.0001)
    loss_fn= nn.BCELoss()
    nepochs=100
    for epoch in range(nepochs):
        for X,y in Xy_loader:
            batch_size= X.shape[0]
            y_hat= model(X.view(batch_size,-1))
            y_hat= y_hat.squeeze()
            loss= loss_fn(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: ",epoch," Loss: ",loss.item())

def task2():
    inputDim = 10
    W_lst=[]
    n=1000
    X = np.random.rand(n,inputDim)
    y= np.random.randint(0,2,n)
    #Layers of Neural Network
    W1= th.tensor(np.random.uniform(0,1,(2,inputDim)),requires_grad=True)
    W2= th.tensor(np.random.uniform(0,1,(3,2)),requires_grad=True)
    W3= th.tensor(np.random.uniform(0,1,3),requires_grad=True)

    W_lst.append(W1)
    W_lst.append(W2)
    W_lst.append(W3)

    loss_fn= nn.BCELoss()
    rn= NeuralNetwork(inputDim)
    #trainNN_sgd(X,y,W_lst,loss_fn,lr=0.0001,nepochs=100)
    #trainNN_batch(X,y,W_lst,loss_fn,lr=0.0001,nepochs=100)
    rn.trainNN_mini_batch(X,y,W_lst,loss_fn,lr=0.0001,nepochs=100,batch_size=10)

def task1():
    inputDim = 10
    n=1000
    X = np.random.rand(n,inputDim)
    y= np.random.randint(0,2,n)
    print(X.shape,y.shape)
    W= th.tensor(np.random.uniform(0,1,inputDim),requires_grad=True)
    print(X[0,:])
    z= wSum(X[0,:],W)
    print(z)

if __name__ == '__main__':
    task6()