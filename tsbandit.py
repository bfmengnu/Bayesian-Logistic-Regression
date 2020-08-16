import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method,Queue
from multiprocessing.dummy import Pool as ThreadPool
import copy
import heapq
from tsbup import vaup, weightup
from scipy.special import comb, perm
from itertools import combinations

# obtain the index of sample with reward 1
def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

# sigma function
def sigma(m):
    return (1 / (1 + torch.exp(-m)))

# encoder subset list as binary list
def encoder(a1,K):
    a = list(np.ones((K,), dtype=int))
    for i in range(K):
        if i not in a1:
            a[i] = int(0)
    return a

def mindex(num_list,topk):
    tmp_list=copy.deepcopy(num_list)
    max_num=sum([abs(O) for O in num_list])
    min_num=-1*max_num
    max_num_index,min_num_index=[],[]
    for i in range(topk):
        one_max_index=num_list.index(max(num_list))
        max_num_index.append(one_max_index)
        num_list[one_max_index]=min_num
    for i in range(topk):
        one_min_index=tmp_list.index(min(tmp_list))
        min_num_index.append(one_min_index)
        tmp_list[one_min_index]=max_num
    return max_num_index

class clickp(nn.Module):
    def __init__(self, duser, daction, heads, K):
        super().__init__()
        #The feature dimension without position
        #M = 1 + d(xt) + d(action)
        self.du = duser
        self.da = daction
        self.heads = heads
        action = torch.randn((self.heads,daction))
        self.action = torch.chunk(action, self.heads, dim=0)
        #self.one = torch.tensor(1).view(1,1).float()
        # size of subset
        self.K = K

    def forward(self, mean, va, xt):
        # This is for parallel computing
        weight = []
        # selected for recommendation
        phi = []
        # K is the size of super-arm
        # mean and va is a list with self.heads vectors corresponding to each arm
        # size of weight vector is M
        # x is context vector
        # Construct the complete weight vector and phi vector
        for i in range(self.heads):
            # Dimension: (1,M)
            phi.append(torch.cat((xt.view(1,-1), self.action[i].view(1,-1)), dim=1))
            # generate a vector with M elements from each distribution
            #print(np.shape(mean[i]))
            #if np.all(np.linalg.eigvals(va[i]) >= 0):
            mu = np.random.multivariate_normal(mean[i], np.linalg.inv(va[i]), (1,), 'raise')
            #else:
            #mu = mean[i]
            # dimension: (M,1)
            w = torch.tensor(mu).view(-1, 1).float()
            weight.append(w)
        # construct click-through probability
        pclick = []
        for i in range(self.heads):
            expectation1 = torch.mm(phi[i],weight[i])
            # Perform logistic link function
            pclick.append(sigma(expectation1))
        subset = mindex(pclick,self.K)
        subset = encoder(subset,self.heads)
        return subset

class classify(nn.Module):
    def __init__(self, duser, daction, heads,K):
        super().__init__()
        # The feature dimension without position
        # M = 1 + d(xt) + d(action)
        self.du = duser
        self.da = daction
        self.heads = heads
        self.M = duser + daction
        self.K = K
        action = torch.randn((self.heads, daction))
        self.action = torch.chunk(action, self.heads, dim=0)
        self.one = torch.tensor(1).view(1, 1).float()
        self.sample_cumu = [[] for _ in range(heads)]

    def forward(self,xt,rewards,subset):
        # rewards is a list
        # This is not for parallel computing
        list_r1 = get_index1(rewards, 1)
        # each heads culmultive sample vectors
        scumu = self.sample_cumu
        xi_1 = []
        for i in range (len(list_r1)):
            xi_1.append(xt[list_r1[i]])
        for index in range(self.heads):
            for i in range(len(list_r1)):
                if subset[list_r1[i]][index] >= 0.1:
                    medium = torch.cat((xi_1[i],self.action[index]),dim=1)
                    scumu[index].append(medium)
        return scumu

class updatemean(nn.Module):
    def __init__(self, duser, daction, heads):
        super().__init__()
        # The feature dimension without position
        # M = 1 + d(xt) + d(action)
        self.du = duser
        self.da = daction
        self.heads = heads
        self.weightup = weightup(duser,daction,heads)
    def forward(self,mean,va,scumu):
        # kmatrix is a complex list, main list has heads,
        # each head list contains all samples corrsponding to the head
        # This is for parallel computing
        mean = self.weightup(mean,va,scumu)
        return mean

class updateva(nn.Module):
    def __init__(self, duser, daction, heads):
        super().__init__()
        # The feature dimension without position
        # M = 1 + d(xt) + d(action)
        self.du = duser
        self.da = daction
        self.heads = heads
        self.vaup = vaup(duser, daction, heads)
    def forward(self,mean,va,scumu):
        # kmatrix is a complex list, main list has heads,
        # each head list contains all samples corrsponding to the head
        # This is for parallel computing
        va = self.vaup(mean,va,scumu)
        return va


