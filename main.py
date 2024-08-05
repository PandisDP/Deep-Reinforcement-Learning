from QDQ  import QDQ
from  Games import Field
from Agent import Agent ,EpsilonGreedyStrategy,ReplayMemory,PrioritizedReplayMemory
import torch as th

def training_process(params_env,prms_tra,type_memory):
    '''
    Params:
    params_env: Dictionary with the parameters of the game
    prms_tra: Dictionary with the parameters of the training
    type_memory: 0 for ReplayMemory and 1 for PrioritizedReplayMemory
    '''
    device= th.device("cuda" if th.cuda.is_available() else "cpu")
    env= Field(device,params_env['size'],params_env['start_position'],params_env['item_pickup'],
                params_env['item_dropoff'],params_env['zones_block'],params_env['Path'])
    eps= EpsilonGreedyStrategy(prms_tra['eps_start'],prms_tra['eps_end'],prms_tra['eps_decay'])
    agent= Agent(eps,env.get_number_of_actions(),device)
    if type_memory==0:
        memory= ReplayMemory(prms_tra['memory_size'])
        q= QDQ(device,env,agent,memory,prms_tra['features'])
        q.training_replay_memory(prms_tra['batch_size'],prms_tra['gamma'],
                            prms_tra['target_update'],prms_tra['lr'],prms_tra['num_episodes'])
    elif type_memory==1:
        memory= PrioritizedReplayMemory(prms_tra['memory_size'])
        q= QDQ(device,env,agent,memory,prms_tra['features'])
        q.training_priorized_memory(prms_tra['batch_size'],prms_tra['gamma'],
                            prms_tra['target_update'],prms_tra['lr'],prms_tra['num_episodes'])
    else:
        print('Error: type_memory must be 0 or 1')
        return 'Error' 
    return 0
if __name__ == '__main__':
    params_game = {
                "size": 10,
                "start_position": (9, 0),  # (9,0)
                "item_pickup": (1, 1),  # (1,1)
                "item_dropoff": (8, 8),  # (8,8)
                "zones_block": [(4, 0), (4, 1), (4, 2), (4, 3), (2, 6), (2, 7), (2, 8), (2, 9), 
                                (4, 8), (5, 8), (6, 8), (7, 6), (8, 6), (9, 6)],
                "Path": 'Episodes'
    }
    params_training = {
                "batch_size": 128,
                "features": 1,
                "gamma": 0.99,
                "eps_start": 1,
                "eps_end": 0.01,
                "eps_decay": 0.001,
                "target_update": 5000,
                "memory_size": 100000,
                "lr": 0.001,
                "num_episodes": 10000
    }
    training_process(params_game,params_training,1)