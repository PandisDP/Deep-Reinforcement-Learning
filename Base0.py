import numpy as np
import torch
import torch.nn as nn
import sys


def wSum(X,W):
    h= torch.from_numpy(X)
    z = torch.matmul(W,h)
    return z 


class NeuralNetwork:
    def __init__(self,inputDim):
        self.inputDim=inputDim

    def forwardStep(self,X,W_list):
        h= torch.from_numpy(X)
        for W in W_list:
            z=torch.matmul(W,h)
            h=self.activate_functions(z)
        return h    
    
    def updateParams(self,W_list,dw_list,lr):
        with torch.no_grad():
            for i in range(len(W_list)):
                W_list[i]-=lr*dw_list[i]

    def activate_functions(self,x):
        return 1/(1+torch.exp(-x))


    def trainNN_sgd(self,X,y,W_list,loss_fn,lr=0.0001,nepochs=1000):
        for epoch in range(nepochs):
            avgLoss=[]
            for i in range(len(y)):
                Xin = X[i,:]
                yin = y[i]
                y_hat= self.forwardStep(Xin,W_list)
                loss= loss_fn(y_hat, torch.tensor(yin,dtype=torch.double))
                loss.backward()
                avgLoss.append(loss.item())
                sys.stdout.flush()
                dW_list=[]  
                for j in range(len(W_list)):
                    dW_list.append(W_list[j].grad.data)
                self.updateParams(W_list,dW_list,lr) 
                for j in range(len(W_list)): 
                    W_list[j].grad.data.zero_()
            print("Epoch: ",epoch," Loss: ",np.mean(np.array(avgLoss)))        

    def trainNN_batch(self,X,y,W_list,loss_fn,lr=0.0001,nepochs=1000):
        n=len(y)
        for epoch in range(nepochs):
            loss=0
            for i in range(n):
                Xin = X[i,:]
                yin = y[i]
                y_hat= self.forwardStep(Xin,W_list)
                loss+= loss_fn(y_hat, torch.tensor(yin,dtype=torch.double))
            loss=loss/n    
            loss.backward()
            sys.stdout.flush()
            dW_list=[]  
            for j in range(len(W_list)):
                dW_list.append(W_list[j].grad.data)
            self.updateParams(W_list,dW_list,lr) 
            for j in range(len(W_list)): 
                W_list[j].grad.data.zero_()
            print("Epoch: ",epoch," Loss: ",loss.item()) 

    def trainNN_mini_batch(self,X,y,W_list,loss_fn,lr=0.0001,nepochs=1000,batch_size=10):
        n=len(y)
        nbatch= n//batch_size
        for epoch in range(nepochs):
            for batch in range(nbatch):
                X_batch= X[batch*batch_size:(batch+1)*batch_size,:]
                y_batch= y[batch*batch_size:(batch+1)*batch_size]
                loss=0
                for i in range(batch_size):
                    Xin = X_batch[i,:]
                    yin = y_batch[i]
                    y_hat= self.forwardStep(Xin,W_list)
                    loss+= loss_fn(y_hat, torch.tensor(yin,dtype=torch.double))
                loss=loss/batch_size    
                loss.backward()
                sys.stdout.flush()
                dW_list=[]  
                for j in range(len(W_list)):
                    dW_list.append(W_list[j].grad.data)
                self.updateParams(W_list,dW_list,lr) 
                for j in range(len(W_list)): 
                    W_list[j].grad.data.zero_()
            print("Epoch: ",epoch," Loss: ",loss.item()/nbatch) 
