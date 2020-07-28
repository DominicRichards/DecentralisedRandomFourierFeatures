import torch
import numpy as np

import networkx as nx
from torch.autograd import Variable
import copy
import time

'''
Functions for simulating Decentralised RFF. 
'''

#Construct Gossip Matrix given number of agents and topology
def GenerateGossipMatrix(NoAgents,Topology):
    if(Topology == "Grid"):
        Graph_Adj = nx.adj_matrix(nx.grid_graph([int(np.sqrt(NoAgents)),int(np.sqrt(NoAgents))],periodic=True)).toarray()
    elif(Topology=="Cycle"):
        Graph_Adj = nx.adj_matrix(nx.grid_graph([NoAgents],periodic=True)).toarray()
    elif(Topology=="Expander"):
        Graph_Adj = nx.adj_matrix(nx.margulis_gabber_galil_graph(int(np.sqrt(NoAgents)))).toarray()
    
    Graph_Gossip = np.identity(NoAgents) -  (np.diag(Graph_Adj.sum(axis=0)) - Graph_Adj )/(Graph_Adj.sum(axis=0).max() + 1)
    if( np.sum(Graph_Gossip.sum(0) != 1.) > 0 or np.sum(Graph_Gossip.sum(1) != 1.)  ):
        print("Gossip Matrix not doubly stochastic! ")
    return(Graph_Gossip)

#Generate Simulated data for testing
def GenerateData(SampleSize,Dimension,NoiseSD):
    X = np.random.normal(size=(SampleSize,Dimension))
    TrueCoef = np.zeros(Dimension) + 1./np.sqrt(Dimension)
    Y = np.sign(X.dot(TrueCoef) + np.random.normal(size=SampleSize)*NoiseSD)
    return(X,Y,TrueCoef)

#Single machine Random Feature Regression with graident descent
def Experiment_GD_RFF(x_train,y_train,x_test,y_test,Iterations,Stepsize,RFFSizes,Replications=1,Sigma =1,device="cpu"):
    TestPerformanceRR = torch.zeros([Replications,Iterations,len(RFFSizes)],dtype=torch.float).to(device)
    
    x_train_tensor = x_train
    x_test_tensor = x_test 

    y_train_tensor  = 2* y_train - 1
    y_test_tensor = 2 * y_test - 1
            
    for i in np.arange(Replications):
        for k,RFFSize in enumerate(RFFSizes):
            
            U = torch.empty([x_train_tensor.shape[1],RFFSize],device=device).normal_(mean=0.,std=1)
            q_vec = torch.rand(RFFSize,device = device)
            
            x_train_transformed = torch.cos(Sigma* torch.matmul(x_train_tensor,U) + q_vec)/np.sqrt(RFFSize)
            x_test_transformed = torch.cos(Sigma *torch.matmul(x_test_tensor,U) + q_vec)/np.sqrt(RFFSize)
            
            x_train_covar = torch.matmul(x_train_transformed.T, x_train_transformed)/x_train_transformed.shape[0]
            
            b = torch.zeros(RFFSize, requires_grad=True, dtype=torch.float, device=device)
            lr = Stepsize
            optimizer = torch.optim.SGD([b], lr=lr)
            
            for j in range(Iterations):
                yhat = torch.matmul(x_train_transformed,b) 
                
                error = y_train_tensor - yhat
                loss = (error ** 2).mean()

                loss.backward()    
                optimizer.step()
                optimizer.zero_grad()
                TestPerformanceRR[i,j,k] = (y_test_tensor* torch.matmul(x_test_transformed,b) < 0 ).float().mean()
    return(TestPerformanceRR)

#Decentralised Random Feature Regression with Distributed Gradient Descent
def Experiment_Decentralised_GD_RF(x_train,y_train,x_test,y_test,Iterations,Stepsize,NoAgents,Topology,RFFSizes,Replications=1,Sigma = 1,device = "cpu"):    
    
    #SplitData
    
    DataSetSizePerAgent = int(x_train.shape[0] / NoAgents)
    DataIndexes = torch.randperm(DataSetSizePerAgent * NoAgents)
    
    #Split the data across agents
    AgentDataIndexes = list(torch.split(DataIndexes,DataSetSizePerAgent))
        
    #Some pre-processing
    x_train_tensor = x_train
    x_test_tensor = x_test 
    y_train_tensor  = 2 * y_train - 1
    y_test_tensor = 2 * y_test - 1
    
    #Construct gossip matrix and put onto device
    GossipMatrix = GenerateGossipMatrix(NoAgents,Topology)
    GossipMatrix_tensor = torch.from_numpy(GossipMatrix).float().to(device)

    TestPerformanceRR = torch.zeros([Replications,Iterations,len(RFFSizes)],dtype=torch.float).to(device)
    
    
    for i in np.arange(Replications):
        for k,RFFSize in enumerate(RFFSizes):

            U = torch.empty([x_train_tensor.shape[1],RFFSize],device=device,dtype=torch.float).normal_(mean=0.,std=1)
            q_vec = torch.rand(RFFSize,device = device)
            
            #Transform the data using Random Fourier Features
            x_train_transformed = torch.cos(Sigma* torch.matmul(x_train_tensor,U) + q_vec)/np.sqrt(RFFSize)
            x_test_transformed = torch.cos(Sigma *torch.matmul(x_test_tensor,U) + q_vec)/np.sqrt(RFFSize)

            #Vectorise the gradient updates across agents using block diagonal matrices. 

            AgentCovarTensor_BlockDiag = torch.zeros((NoAgents*RFFSize,NoAgents*RFFSize),dtype=torch.float,device=device)
            AgentXYTensor_Block = torch.zeros((NoAgents*RFFSize),dtype=torch.float,device=device)
            for j in np.arange(NoAgents):
                Agent_x_train = x_train_transformed[AgentDataIndexes[j]]
                Agent_y_train = y_train_tensor[AgentDataIndexes[j]]
                
                AgentCovarTensor_BlockDiag[j*RFFSize : (j+1) * RFFSize, j*RFFSize : (j+1) * RFFSize] = torch.matmul(Agent_x_train.T, Agent_x_train)/Agent_x_train.shape[0]
                AgentXYTensor_Block[j*RFFSize : (j+1)*RFFSize] = torch.matmul(Agent_x_train.T,Agent_y_train)

            #Where to store parameter
            b = torch.zeros([NoAgents,RFFSize],requires_grad=False, dtype=torch.float, device=device)
            
            lr = Stepsize 
            #Start gradient descent iterations
            for j in np.arange(Iterations):

                b = b.reshape(NoAgents * RFFSize)
                b = b - lr *torch.matmul(AgentCovarTensor_BlockDiag,b) + lr * AgentXYTensor_Block
                b = b.reshape(NoAgents, RFFSize)
                b = torch.matmul(GossipMatrix_tensor,b)
                TestPerformanceRR[i,j,k] = ( (y_test_tensor * torch.mm(x_test_transformed,b.T).T).T < 0).float().mean(axis=0).max()

    return(TestPerformanceRR)
