import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigma(m):
    return (1 + torch.exp(-m))

def upq(w,x1,q,bs):
    s = 0
    for i in range(bs):
        p = sigma(torch.mm(x1[i],w))
        s = s + p*(1-p) * torch.mm(x1[i].transpose(0,1), x1[i])
    q = s + q
    return q

def upw(w,q,m,x1,bs):
    m = torch.unsqueeze(m, dim=1)
    aa = w - m
    inter1 = torch.mm(aa.transpose(0,1), q)
    inter1 = 0.5 * torch.mm(inter1, aa)
    inter2 = 0
    for i in range(bs):
        inter2 = inter2 + torch.log(sigma(torch.mm(x1[i],m)))
    return (inter1 + inter2)

class weightup(nn.Module):
    def __init__(self, duser, daction, heads):
        super().__init__()
        # The number of avaiable actions(maximum layers)
        self.M = duser + daction
        # the number of iteration for gradient descent
        self.num = 200
    def forward(self, mean, va, xt):
        mean = torch.from_numpy(mean).float()
        va = torch.from_numpy(va).float()
        initialize = torch.ones((self.M,1)).float()
        w = initialize.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w, ], lr=0.1)
        bs = len(xt)
        #min = []
        if bs>=0.1:
            for step in range(self.num):
                if step:
                    optimizer.zero_grad()
                    f.backward(retain_graph=True)
                    optimizer.step()
                f = upw(w, va, mean, xt, bs)
                #min.append(w.detach().numpy())
            mean = w.detach().numpy()
            mean = np.squeeze(mean)
            #if mean.all() == min[-1].all():
            #    print("yes")
        else:
            mean = mean.numpy()
        return mean

class vaup(nn.Module):
    def __init__(self, duser, daction, heads):
        super().__init__()
        # The number of avaiable actions(maximum layers)
        self.daction = daction
    def forward(self, mean, va, xt):
        bs = len(xt)
        mean = torch.from_numpy(mean).float()
        mean = torch.unsqueeze(mean,dim=1)
        va = torch.from_numpy(va).float()
        if bs >= 0.1:
            va = upq(mean, xt, va, bs)
        else:
            va = va
        va = va.numpy()
        return va





