#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
import torch
def volume_prism_x(M):
    return torch.sum(M[:,:,:,0],dim=2)*(torch.linalg.det(M[:,:,1:,1:]-M[:,:,0,1:].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_y(M):
    return torch.sum(M[:,:,:,1],dim=2)*(torch.linalg.det(M[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])],indexing="ij")[1]]-M[:,:,1,[0,2]].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_z(M):
    return torch.sum(M[:,:,:,2],dim=2)*(torch.linalg.det(M[:,:,:2,:2]-M[:,:,2,:2].reshape(M.shape[0],M.shape[1],1,-1))/6)


def volume_2_x(mesh):
    return torch.sum(volume_prism_x(mesh),dim=1)

def volume_2_y(mesh):
    return torch.sum(volume_prism_y(mesh),dim=1)



def volume_2_z(mesh):
    return torch.sum(volume_prism_z(mesh),dim=1)


def get_coeff_z(points,triangles,points_triangles_matrix):
    tmp=points[:,triangles]
    tmp1=torch.linalg.det(tmp[:,:,:2,:2]-tmp[:,:,2,:2].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    return tmp1@points_triangles_matrix

def get_coeff_x(points,triangles,points_triangles_matrix):
    tmp=points[:,triangles]
    tmp1=torch.linalg.det(tmp[:,:,1:,1:]-tmp[:,:,0,1:].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    return tmp1@points_triangles_matrix

def get_coeff_y(points,triangles,points_triangles_matrix):
    tmp=points[:,triangles]
    tmp1=torch.linalg.det(tmp[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])],indexing="ij")[1]]-tmp[:,:,1,[0,2]].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    return tmp1@points_triangles_matrix

def lin_solve(A,b):
    return torch.bmm(torch.bmm(torch.transpose(A,1,2),torch.inverse(torch.bmm(A,torch.transpose(A,1,2)))),b)

class Double(nn.Module):
    def __init__(self,batch_size,triangles,barycenter,volume,points_triangles_matrix):
        super().__init__()
        self.barycenter=barycenter
        self.volume=volume
        self.points_triangles_matrix=points_triangles_matrix
        self.triangles=triangles
        self.batch_size=batch_size

    def forward(self,x):
        x=x.reshape(x.shape[0],-1,3)
        bar=torch.mean(x,axis=1)
        volume=volume_2_x(x[:,self.triangles])
        y=x.clone()
        ax=self.barycenter[0].unsqueeze(0).repeat(x.shape[0])-bar[:,0]
        ay=self.barycenter[1].unsqueeze(0).repeat(x.shape[0])-bar[:,1]
        az=self.barycenter[2].unsqueeze(0).repeat(x.shape[0])-bar[:,2]
        ax=ax.reshape(-1,1,1)
        ay=ay.reshape(-1,1,1)
        az=az.reshape(-1,1,1)
        a=1/3*(self.volume-volume).reshape(-1,1,1)
        Avx=get_coeff_x(y,self.triangles,self.points_triangles_matrix).unsqueeze(1)
        B=torch.ones_like(Avx)/x.shape[1]
        Ax=torch.concatenate((Avx,B),dim=1)
        bx=torch.concatenate((a,ax),dim=1)
        def_x=lin_solve(Ax,bx)
        y[:,:,0]=x[:,:,0]+def_x.squeeze(2)
        Avy=get_coeff_y(y,self.triangles,self.points_triangles_matrix).unsqueeze(1)
        Ay=torch.concatenate((Avy,B),dim=1)
        by=torch.concatenate((a,ay),dim=1)
        def_y=lin_solve(Ay,by)
        y[:,:,1]=x[:,:,1]+def_y.squeeze(2)
        Avz=get_coeff_z(y,self.triangles,self.points_triangles_matrix).unsqueeze(1)
        Az=torch.concatenate((Avz,B),dim=1)
        bz=torch.concatenate((a,az),dim=1)
        def_z=lin_solve(Az,bz)
        y[:,:,2]=x[:,:,2]+def_z.squeeze(2)
        return y

