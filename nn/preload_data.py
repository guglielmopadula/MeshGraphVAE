#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
import numpy as np
import meshio
import qllr
from tqdm import trange
def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    return points
NUM_SAMPLES=600
points=getinfo("data/bunny_0.ply")
a=np.zeros((NUM_SAMPLES,points.reshape(-1).shape[0]-3))
tmp=getinfo("data/bunny_{}.ply".format(0))
bar=np.mean(tmp,axis=0)
num_points=len(tmp)

def matrix(x):
    A=np.tile(np.eye(3),num_points).reshape(1,3,num_points)
    A=np.tile(A,(x.shape[0],1,1))
    return A

lin_indices=[i for i in range(3*num_points)]

q=qllr.QLLR(matrix,bar,lin_indices,[])

for i in trange(NUM_SAMPLES):
    a[i]=q.transform(getinfo("data/bunny_{}.ply".format(i)).reshape(1,-1)).reshape(-1)


np.save("nn/transformer.npy",q)
np.save("data/data.npy",a)
