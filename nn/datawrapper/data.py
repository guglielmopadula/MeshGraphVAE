#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import torch
import meshio
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy
from scipy.sparse import coo_array
from models.basic_layers.PCA import PCA
from torch.utils.data import random_split
torch.set_default_dtype(torch.float32)


def scipy_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def volume_2_x(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,0],axis=2)*(np.linalg.det(mesh[:,:,1:,1:]-np.expand_dims(mesh[:,:,0,1:],2))/6),axis=1)
    return tmp.reshape(shape[:-3])

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    barycenter=np.mean(points,axis=0)
    triangles=mesh.cells_dict["triangle"]
    volume=volume_2_x(points[triangles])
    n_points=len(points)
    n_triangles=len(triangles)
    l=[]
    for i in range(n_triangles):
        for j in triangles[i]:
            l.append([i,j])
    l=np.array(l)
    row=l.T[0]
    col=l.T[1]
    data=np.ones(len(col),dtype=np.float32)
    arr=coo_array((data, (row, col)), shape=(n_triangles, n_points))
    return torch.tensor(points),torch.tensor(barycenter),torch.tensor(volume),torch.tensor(triangles),scipy_to_torch(arr)


class Data(LightningDataModule):
    def get_size(self):
        return ((1,self.n_points,3))
    
    def get_reduced_size(self):
        return self.reduced_dimension

    def __init__(
        self,batch_size,num_workers,num_train,num_test,reduced_dimension,string,use_cuda,pool_vec_names,pool_adj_names):
        super().__init__()
        self.l=[]
        self.batch_size=batch_size
        self.adj=[]
        for name in pool_vec_names:
            self.l.append(torch.tensor(np.load(name),dtype=torch.long))
        for name in pool_adj_names:
            self.adj.append(torch.tensor(np.load(name),dtype=torch.long).T)
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_train=num_train
        self.num_workers = num_workers
        self.num_test=num_test
        self.reduced_dimension=reduced_dimension
        self.string=string
        self.num_samples=self.num_test+self.num_train
        tmp,self.barycenter,self.volume,self.triangles,self.points_triangles_matrix=getinfo(string.format(0))
        self.n_points=tmp.shape[0]

        self.data=torch.zeros(self.num_samples,*tmp.shape)
        for i in range(0,self.num_samples):
            if i%100==0:
                print(i)
            self.data[i],_,_,_,_=getinfo(self.string.format(i))
        self.pca=PCA(self.reduced_dimension)
        if use_cuda:
            self.pca.fit(self.data.reshape(self.num_samples,-1).cuda())
            self.barycenter=self.barycenter.cuda()
        else:
            self.pca.fit(self.data.reshape(self.num_samples,-1))

        self.data_train,self.data_test = random_split(self.data, [self.num_train,self.num_test])    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
