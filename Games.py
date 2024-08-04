import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import torch
class Field:
    def __init__(self,device,size,item_pickup,item_dropoff,
                start_position,zones_blocks=[],path_predicts='Episodes'):
        '''
        Constructor of the class Field
        Parameters:
        size: size of the field
        item_pickup: position of the item to pickup
        item_dropoff: position of the item to dropoff
        start_position: position of the agent
        zones_blocks: list of tuples with the positions of the blocks
        path_predicts: path to save the images of the episodes
        '''
        self.device=device
        self.size = size
        self.item_pickup = item_pickup
        self.item_dropoff = item_dropoff
        self.position = start_position
        self.position_start= start_position
        self.block_zones=zones_blocks
        self.item_in_car= False
        self.number_of_actions=6
        self.allposicions = []
        self.path_predicts= path_predicts
        self.done=False
        self.save_state()
        self.initial_state = {
            'device': self.device,
            'position': self.position,
            'item_pickup': self.item_pickup,
            'item_dropoff': self.item_dropoff,
            'item_in_car': self.item_in_car
        }

    def reset(self):
        '''
        Reset the game
        '''
        self.device = self.initial_state['device']
        self.position = self.initial_state['position']
        self.item_pickup = self.initial_state['item_pickup']
        self.item_dropoff = self.initial_state['item_dropoff']
        self.item_in_car = self.initial_state['item_in_car']
        self.done=False
        self.allposicions = []
        self.save_state()    

    def get_number_of_actions(self):
        '''
        Get the number of actions of the game
        Returns: number of actions
        '''
        return self.number_of_actions
    
    def get_number_of_states(self):
        '''
        Get the number of states of the game
        Returns: number of states
        '''
        return (self.size**4)*2 

    def get_state(self):
        '''
        Get the state of the game
        Returns: state
        '''
        state= self.position[0]*self.size*self.size*self.size*2
        state+= self.position[1]*self.size*self.size*2
        state+= self.item_pickup[0]*self.size*2
        state+= self.item_pickup[1]*2   
        if self.item_in_car:
            state+=1
        return torch.tensor([state],device=self.device)   
    
    def save_state(self):
        '''
        Save the state of the game
        '''
        self.allposicions.append(self.position)

    def graphics(self,puntos,name_fig):
        '''
        Create a plot of the game
        Parameters:
        puntos: list of tuples with the positions of the points
        name_fig: name of the figure
        '''
        # Crear una cuadrícula de 10x10
        cuadricula = np.zeros((10, 10))
        # Marcar los puntos en la cuadrícula
        for punto in puntos:
            cuadricula[punto] = 1
        # Crear la figura y el eje para el plot
        fig, ax = plt.subplots()
        # Usar 'imshow' para mostrar la cuadrícula como una imagen
        # 'cmap' define el mapa de colores, 'Greys' es bueno para gráficos en blanco y negro
        ax.imshow(cuadricula, cmap='Greys', origin='lower')
        # Ajustar los ticks para que coincidan con las posiciones de la cuadrícula
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        # Dibujar las líneas de la cuadrícula
        ax.grid(color='black', linestyle='-', linewidth=2)
        # Ajustar el límite para evitar cortes
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        for punto in self.block_zones:
            ax.scatter(punto[1], punto[0], color='red', marker='X', s=100) 
        for punto in puntos:
            ax.text(punto[1], punto[0], '✔', color='white', ha='center', va='center', fontsize=10)

        lst_start=[self.position_start, self.item_pickup,self.item_dropoff]
        for punto in lst_start:
            ax.scatter(punto[1], punto[0], color='blue',marker='*', s=100)  
        name_fig_path = self.path_predicts + '/' +name_fig
        plt.savefig(name_fig_path)
        plt.close()

    def empty_predict_data(self):
        '''
        Empty the folder of the predictions
        '''
        path=self.path_predicts
        for nombre in os.listdir(path):
            ruta_completa = os.path.join(path, nombre)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.remove(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f'Error {ruta_completa}. reason: {e}')

    def block_zones_evaluation(self,position):
        '''
        Evaluate if the position is in a block zone
        Parameters:
        position: position to evaluate
        Returns: True if the position is in a block zone, False otherwise
        '''
        if position in self.block_zones:
            return True
        return False

    def make_action(self,action):
        '''
        Make an action in the game
        Parameters:
        action: action to make
        Returns: reward, done
        '''
        val_return=0
        (x,y) = self.position
        if action ==0: #down
            if y==self.size-1:
                val_return= -10 #reward,done
                return torch.tensor([val_return],device=self.device),self.done
            else:
                self.position = (x,y+1)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    val_return= -100
                    return torch.tensor([val_return],device=self.device),self.done 
                val_return = -1
                return torch.tensor([val_return],device=self.device),self.done
        elif action ==1: #up
            if y==0:
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done  
            else:
                self.position = (x,y-1)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    val_return =-100
                    return torch.tensor([val_return],device=self.device),self.done  
                val_return = -1
                return torch.tensor([val_return],device=self.device),self.done
        elif action ==2: #left
            if x==0:
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done  
            else:
                self.position = (x-1,y)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    val_return = -100
                    return torch.tensor([val_return],device=self.device),self.done  
                val_return= -1
                return torch.tensor([val_return],device=self.device),self.done  
        elif action ==3: #right
            if x==self.size-1:
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done  
            else:
                self.position = (x+1,y)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    val_return =-100
                    return torch.tensor([val_return],device=self.device),self.done  
                val_return = -1
                return torch.tensor([val_return],device=self.device),self.done 
        elif action ==4: #pickup
            if self.item_in_car:
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done   
            elif self.item_pickup != (x,y):
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done  
            else:
                self.item_in_car = True
                val_return = 20
                return torch.tensor([val_return],device=self.device),self.done
        elif action ==5: #dropoff
            if not self.item_in_car:
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done  
            elif self.item_dropoff != (x,y):
                val_return = -10
                return torch.tensor([val_return],device=self.device),self.done   
            else:
                self.item_in_car = False
                self.done=True
                val_return = 20
                return torch.tensor([val_return],device=self.device),self.done  


