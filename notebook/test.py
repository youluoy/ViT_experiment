import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F



#%%
a = torch.ones(2, 3, 4, 4)
su = torch.sum(a, 1)
print(su.size())
print(su.unsqueeze(dim=1).size())


print(a.unfold(2, 2, 2).size())
print(a.unfold(3, 2, 2).size())


b = torch.ones(2, 4, 4, 4)

inputs = torch.ones(2,3,4,4)
filters = torch.ones(3,3,2,2)
F.conv2d(inputs, filters, padding=0)
filters[:,1,:,:] = torch.ones(2,2) + torch.ones(2,2)
filters[:,2,:,:] = torch.ones(2,2) + torch.ones(2,2) + torch.ones(2,2)
filters

c = F.conv2d(inputs, filters, padding=0)
print(c.size())
#print(c)
#%%
e = torch.zeros(2, 3, 4, 4)
r = 0
for i in range(2):
    for j in range(3):
        for k in range(4):
            for l in range(4):
                a[i,j,k,l] = r
                r += 1

d = e.unfold(2, 2, 2).unfold(3, 2, 2)
print(d.size())
#%%
f = d.reshape(2, 3, -1, 2, 2)
print(f.size())
print(f)
g = f.permute(0, 2, 1, 3, 4)
print(g.size())
print(g)

print(inputs.size())
img = inputs.unsqueeze(dim=1)
print(img.size())
print(g.size())
#%%
print("inputs.{}".format(inputs[0].unsqueeze(dim=0).size()))
window = g[0]
print("window.{}".format(window.size()))
h = F.conv2d(inputs[0].unsqueeze(dim=0), window, padding=0)
print("h.{}".format(h.size()))
print(h)
# %%
print(h.size()[0])
n = torch.zeros_like(h)
m = n.size()
#m[0] = 10
print(m)

hh = torch.zeros(1,2,2)
ww = torch.zeros(1,2,2)
bb = []
bb.append(hh)
bb.append(ww)
cc = torch.cat(bb, dim = 0)
print(cc.size())

#%%
torch.cuda.is_available()
x = torch.zeros(1, 2, 2)
y = x.copy()
y[:,:,:] = 1
print(x)

#%%
device = 'cuda'
x = torch.tensor([2.0], device=device, requires_grad=False)
w = torch.tensor([1.0], device=device, requires_grad=True)
b = torch.tensor([3.0], device=device, requires_grad=True)

y = x*w + b
y.backward()

z = w.detach()
s = z.clone()

print(z)
print(z.grad)
print(z is w)
print(z.is_leaf)

print(s)
print(s.grad)
print(s is z)
print(s.is_leaf)
#%%


aa = torch.ones(2, 2, 2, 2)
aa = torch.flatten(aa, start_dim=2)
aa.size()
m = nn.ReLU()
print(m(aa))

#%%
l = 0
cc = torch.zeros(2, 3, 4, 2, 2)
for i in range(cc.size()[0]):
    for j in range(cc.size()[1]):
        for k in range(cc.size()[2]):  
            cc[i,j,k,:] = l
            l += 1
#print(cc)
cc = cc.permute(0, 1, 3, 4, 2)
cc = cc.reshape(2, 3*2*2, 4)
fold = nn.Fold(output_size=(4, 4), kernel_size=(2,2))
cc = fold(cc)
print(cc)


#%%

l=0
cc = torch.zeros(2, 3, 4, 2, 2)
for i in range(cc.size()[0]):
    for j in range(cc.size()[1]):
        for k in range(cc.size()[2]):  
            cc[i,j,k,:] = l
            l += 1
cc = cc.permute(0, 2, 1, 3, 4)
dd = torchvision.utils.make_grid(cc[0], nrow=2, padding=0)
print(dd)
dd.size()


#%%
li = nn.Linear(10, 20)
ab = torch.ones(2, 3, 4, 10)
cd = li(ab)
print(cd.size())


#%%
a = torch.zeros(2,1,3,4)
b = torch.ones(2,1,3,4)
c = -torch.ones(2,1,3,4)
d = []
d.append(a)
d.append(b)
d.append(c)
d = torch.cat(d, dim=1)
print(d.size())
print(d)