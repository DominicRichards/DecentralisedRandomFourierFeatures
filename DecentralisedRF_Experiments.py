from DecentralisedRF_Functions import *

import torch
import numpy as np
import networkx as nx
from torch.autograd import Variable


import matplotlib.pyplot as plt
import copy
import os 


'''
Code for implementing Decentralised Random Fourier Features with Distributed Graident Descent. 
Implementation utilises Pytorch for GPU implementation of matrix-vector products.  
'''


'''
Import SUSYDATA
'''
print("Loading SUSY ...")
SUSYDATA = np.loadtxt("../Datasets/SUSY_csv",delimiter=",")

GPUNumber = 0
SimCount = 3 #Experiment number
device = "cuda:" + str(GPUNumber)
Iterations = 10000
StepSize = 1. 
RFFSizes = [300]
Sigma = 10**(-0.5) #Scaling of variance for Random Features
TestSize =  (10**(4));

TrainSampleSizeTest = (10**(np.linspace(2,4,10))).astype("int");
OuterReplications = 10 #Number of replications for sub-sampling SUSY 
Replications = 5 #Number of replications for averaging out Random Fourier Error
Stepsize = 1.
AgentTests = [25,49,100] #Network sizes


TestErrorStore = torch.zeros([ len(AgentTests)+1,OuterReplications,len(TrainSampleSizeTest)],dtype=torch.float,device=device)

NumberIterationsStore_ArgMinMean = torch.zeros([ len(AgentTests)+1,OuterReplications,len(TrainSampleSizeTest)],dtype=torch.float,device=device)
Topology = "Grid"

TestDir = "GPU" + str(GPUNumber) + "_Sim" + str(SimCount) + "/"
os.mkdir(TestDir)
os.mkdir(TestDir + "Plots/")
os.mkdir(TestDir + "Plots/" + Topology + "/")
os.mkdir(TestDir + "ExperimentalResults/")
os.mkdir(TestDir + "ExperimentalResults/" + Topology + "/")



print("Loading SUSY onto GPU..")
SUSYDATA_Tensor = torch.torch.from_numpy(SUSYDATA).float().to(device)

for k,TrainSize in enumerate(TrainSampleSizeTest):
    print("TrainDataSize = " + str(TrainSize) + " ( " + str(k+1) + " / " + str(len(TrainSampleSizeTest)) + " ) " )

    for l in np.arange(OuterReplications):
        print("Replication " + str(l) + " / " + str(OuterReplications) )
        SubSetSize = TrainSize + TestSize

        #Sub-sample SUSY dataset
        SUSYDATA_Subset = SUSYDATA_Tensor[torch.randperm(SUSYDATA_Tensor.shape[0])[0:SubSetSize]]
        SUSYDATA_Train  = SUSYDATA_Subset[0:TrainSize]
        SUSYDATA_Test = SUSYDATA_Subset[(TrainSize+1):]

        #Perform single-machine Graident Descent
        RFF_GD = Experiment_GD_RFF(SUSYDATA_Train[:,1:],SUSYDATA_Train[:,0],SUSYDATA_Test[:,1:],SUSYDATA_Test[:,0],Iterations,Stepsize,RFFSizes,Replications,Sigma,device=device)
        #Store some output quantities
        TestErrorStore[0,l,k] = RFF_GD.mean(axis=0).min(0)[0]
        NumberIterationsStore_ArgMinMean[0,l,k] = RFF_GD.argmin(axis=1).float().mean(0)[0]

        #Perform Distributed Gradient Descent for different number of agents
        for i,NoAgents in enumerate(AgentTests):
            print("Agents = " + str(NoAgents))
            RFF_GD_Decentralised = Experiment_Decentralised_GD_RF(SUSYDATA_Train[:,1:],SUSYDATA_Train[:,0],SUSYDATA_Test[:,1:],SUSYDATA_Test[:,0],Iterations,Stepsize,NoAgents,Topology,RFFSizes,Replications,Sigma,device=device)
            TestErrorStore[i+1,l,k] = RFF_GD_Decentralised.mean(axis=0).min(0)[0]
            NumberIterationsStore_ArgMinMean[i+1,l,k] = RFF_GD_Decentralised.argmin(axis=1).float().mean(0)[0]

#Move results back over to cpu 
TestErrorStore = TestErrorStore.cpu()
NumberIterationsStore_ArgMinMean = NumberIterationsStore_ArgMinMean.cpu()



#Start Plotting
plt.errorbar(TrainSampleSizeTest,TestErrorStore[0].mean(axis=0),TestErrorStore[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,TestErrorStore[i+1].mean(axis=0),TestErrorStore[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")
# plt.xscale("log")
# plt.yscale("log")
plt.title("Classification Error vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Classification Error")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/" + "ClassficiationVsSampleSize" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()

plt.errorbar(TrainSampleSizeTest,TestErrorStore[0].mean(axis=0),TestErrorStore[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,TestErrorStore[i+1].mean(axis=0),TestErrorStore[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")

plt.xscale("log")
plt.title("Classification Error vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Classification Error")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/"  + "ClassficiationVsSampleSize_xlog" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()

plt.errorbar(TrainSampleSizeTest,TestErrorStore[0].mean(axis=0),TestErrorStore[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,TestErrorStore[i+1].mean(axis=0),TestErrorStore[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")

plt.xscale("log")
plt.yscale("log")
plt.title("Classification Error vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Classification Error")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/" + "ClassficiationVsSampleSizeloglog" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()


plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[0].mean(axis=0),NumberIterationsStore_ArgMinMean[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[i+1].mean(axis=0),NumberIterationsStore_ArgMinMean[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")


# plt.xscale("log")
# plt.yscale("log")
plt.title("Optimal Stopping vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Optimal Stopping Time")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/" + "OptimalStoppingVsSampleSize_ArgMinMean" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()

plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[0].mean(axis=0),NumberIterationsStore_ArgMinMean[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[i+1].mean(axis=0),NumberIterationsStore_ArgMinMean[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")


plt.xscale("log")
# plt.yscale("log")
plt.title("Optimal Stopping vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Optimal Stopping Time")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/" + "OptimalStoppingVsSampleSize_ArgMinMeanxlog" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()

plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[0].mean(axis=0),NumberIterationsStore_ArgMinMean[0].std(axis=0),label="Single Machine")
for i,NoAgents in enumerate(AgentTests):
    plt.errorbar(TrainSampleSizeTest,NumberIterationsStore_ArgMinMean[i+1].mean(axis=0),NumberIterationsStore_ArgMinMean[i+1].std(axis=0),label="Decentralised " + Topology  + " " +r"$(n=$" + str(NoAgents) + r"$)$")


plt.xscale("log")
plt.yscale("log")
plt.title("Optimal Stopping vs Sample Size " + r"($M=300$)" )
plt.xlabel("Total Sample Size " + r"$nm$")
plt.ylabel("Optimal Stopping Time")
plt.legend()
plt.savefig(TestDir + "Plots/" + Topology + "/" + "OptimalStoppingVsSampleSize_ArgMinMeanloglog" + ".pdf",dpi=128,bbox_inches="tight",pad_inches = 0)
plt.close()

np.savetxt(TestDir +  "ExperimentalResults/"+ Topology + "/"  + "TestErrorStore.csv",TestErrorStore.reshape(TestErrorStore.shape[0],TestErrorStore.shape[1]*TestErrorStore.shape[2]),delimiter=",")
np.savetxt(TestDir +  "ExperimentalResults/"+ Topology + "/"  + "NumberIterationsStore_ArgMinMean.csv",NumberIterationsStore_ArgMinMean.reshape(NumberIterationsStore_ArgMinMean.shape[0],NumberIterationsStore_ArgMinMean.shape[1]*NumberIterationsStore_ArgMinMean.shape[2]),delimiter=",")
np.savetxt(TestDir +  "ExperimentalResults/"+ Topology + "/"  + "Dimenions.csv",TestErrorStore.shape,delimiter=",")


print( "Complete ServerTest.py !" )

# DecentralisedTest,b_out = Experiment_Decentralised_GD_RF(x_train,y_train,x_test,y_test,Iterations = Iterations,Stepsize = StepSize,NoAgents = NoAgents,Topology = Topology,RFFSizes = RFFSizes,Sigma=Sigma,device= device)
# plt.plot(DecentralisedTest[0,:,0])
# plt.show()




