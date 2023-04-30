#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
import numpy as np
import meshio
from tqdm import trange
def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    return points

points=getinfo("data/bunny_0.ply")
a=np.zeros((600,points.reshape(-1).shape[0]))
for i in trange(600):
    a[i]=getinfo("data/bunny_{}.ply".format(i)).reshape(-1)
np.save("data/data.npy",a)
