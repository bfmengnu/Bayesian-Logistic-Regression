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
from tsbandit import clickp,classify, updatemean, updateva
import argparse
import time
import torch
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def variance(M,up):
    a = abs(2 * np.random.randn(M, M) + up)
    #a = np.eye(M)
    conva = np.dot(a, a.T)
    return conva

def min_max_scaler1(x):
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train_model(model, opt):
    batch_size = 5
    #min_max_scaler = MinMaxScaler()
    M = opt.duser + opt.daction
    mean100 = (np.ones((M,1)) + 1e-1)
    mean100 = np.squeeze((mean100))
    mean0 = (np.ones((M,1))+ 2e-1)
    mean0 = np.squeeze((mean0))
    mean_100 = (np.ones((M,1)) + 3e-1)
    mean_100 = np.squeeze((mean_100))
    va = []
    va.append((variance(M, 2e-1)))
    va.append((variance(M, 3e-1)))
    va.append((variance(M, 5e-1)))
    mean = []
    mean.append(mean100)
    mean.append(mean0)
    mean.append(mean_100)
    batch_num = 3
    xt = np.ones((opt.duser,),dtype=float)
    xtcovariance = variance(opt.duser,0.5)
    xt = np.random.multivariate_normal(xt, xtcovariance, (batch_size*batch_num,), 'raise')
    xt = torch.from_numpy(xt).float()
    # xt is a list now
    xt = torch.chunk(xt, batch_num, dim=0)
    cumu_rewards = []
    for epoch in range(opt.epochs):
        epoch_reward = 0
        for batch in range(batch_num):
            src = xt[batch]
            mean,va,current_reward = model(src, mean, va)
            epoch_reward = +current_reward
        cumu_rewards.append(epoch_reward)
        print(cumu_rewards)
        plt.figure()
        plt.plot(cumu_rewards)
        plt.savefig("Rewards.png")

class TSbandit(nn.Module):
    def __init__(self, duser, daction, heads, K,action):
        super().__init__()
        self.du = duser
        self.da = daction
        self.heads = heads
        self.M = duser + daction
        self.K = K
        self.clickp = clickp(duser, daction, heads, K,action)
        self.classify = classify(duser, daction, heads,K,action)
        self.updatemean = updatemean(duser, daction, heads)
        self.updateva = updateva(duser, daction, heads)

    def forward(self, xt,mean1,va1):
        # mean and va is a list corresponding to each sample
        mean = []
        va = []
        size = xt.size(0)
        xt = torch.chunk(xt,size,dim=0)
        for i in range(size):
            mean.append(mean1)
            va.append(va1)
        num_cores = mp.cpu_count()
        pool1 = ThreadPool(num_cores)
        subset = pool1.starmap(self.clickp, zip(mean, va, xt))
        pool1.close()
        rewards = []
        # Only former 2 heads are selected, the rewards is 1
        for i in range(len(subset)):
            if subset[i][0]*subset[i][1] > 0:
                rewards.append(int(1))
            else:
                rewards.append(int(0))
        current_reward = np.sum(np.array(rewards))
        scumu = self.classify(xt,rewards,subset)
        pool2 = ThreadPool(num_cores)
        mean1 = pool2.starmap(self.updatemean, zip(mean1, va1, scumu))
        pool2.close()
        pool3 = ThreadPool(num_cores)
        va1 = pool3.starmap(self.updateva, zip(mean1,va1, scumu))
        pool3.close()
        return mean1,va1,current_reward

def get_model(opt):
    model = TSbandit(opt.duser, opt.daction, opt.heads, opt.K,opt.action)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-daction', type=int,default=3)
    parser.add_argument('-duser', type=int,default=16)
    parser.add_argument('-heads', type=int,default=3)
    parser.add_argument('-K', type=int,default=2)
    parser.add_argument('-epochs', type=int, default=100)
    opt = parser.parse_args()
    opt.action = torch.eye(opt.heads)
    model = get_model(opt)
    train_model(model,opt)

if __name__ == "__main__":
    main()


