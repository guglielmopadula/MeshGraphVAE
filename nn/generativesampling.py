#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import torch
from datawrapper.data import Data

from training_full import GAE
import qllr
import matplotlib.pyplot as plt
import torch
import dill
from tqdm import trange
import numpy as np
import meshio
from models.losses.losses import relmmd
cuda_avail=True if torch.cuda.is_available() else False
torch.use_deterministic_algorithms(True)
from pytorch_lightning.plugins.environments import SLURMEnvironment

class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return

def area(vertices, triangles):
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1) / 2
    return np.sum(a)

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    return points

tmp=getinfo("data/bunny_{}.ply".format(0))
bar=np.mean(tmp,axis=0)
num_points=len(tmp)


def matrix(x):
    A=np.tile(np.eye(3),num_points).reshape(1,3,num_points)
    A=np.tile(A,(x.shape[0],1,1))
    return A


NUM_WORKERS = int(os.cpu_count() / 2)

LATENT_DIM_1=15
LATENT_DIM_2=1
NUM_TRAIN_SAMPLES=300
REDUCED_DIMENSION=36
NUM_TEST_SAMPLES=0
BATCH_SIZE = 1
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES


with open("nn/saved_models/bayesdisc.pt", 'rb') as in_strm:
    disc = dill.load(in_strm)

disc.eval()
d={
  GAE: "GAE", 
  }

data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/data.npy"),
          use_cuda=False)
pca=data.pca                           
tets=np.load("data/tetras.npy")
def list_faces(t):
  t.sort(axis=1)
  n_t, m_t= t.shape 
  f = np.empty((4*n_t, 3) , dtype=int)
  i = 0
  for j in range(4):
    f[i:i+n_t,0:j] = t[:,0:j]
    f[i:i+n_t,j:3] = t[:,j+1:4]
    i=i+n_t
  return f

def extract_unique_triangles(t):
  _, indxs, count  = np.unique(t, axis=0, return_index=True, return_counts=True)
  return t[indxs[count==1]]

def extract_surface(t):
  f=list_faces(t)
  f=extract_unique_triangles(f)
  return f

triangles=extract_surface(tets)
data=data.data[:].cpu().numpy().reshape(NUMBER_SAMPLES,-1)
moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))
area_data=np.zeros(NUMBER_SAMPLES)
_=0
lin_indices=[i for i in range(3*num_points)]

transformer=qllr.QLLR(matrix,bar,lin_indices,[])
print(data.reshape(NUMBER_SAMPLES,-1,3).shape)
data2=data.copy()
data2=transformer.inverse_transform(data2)
data2=data2.reshape(NUMBER_SAMPLES,-1,3)



#print(np.mean(data2,axis=1))
#data2=data2-np.mean(data2,axis=1).reshape(NUMBER_SAMPLES,1,3).repeat(data2.shape[1],axis=1)
for j in range(3):
    for k in range(3):
        moment_tensor_data[:,j,k]=np.mean(data2.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*data2.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)

for i in trange(NUMBER_SAMPLES):
    area_data[i]=area(data2[i],triangles)

for wrapper, name in d.items():
    area_sampled=np.zeros(NUMBER_SAMPLES)
    torch.manual_seed(100)
    np.random.seed(100)
    temp=np.zeros(data.shape)
    model=GAE()
    model=torch.load("./nn/saved_models/"+name+".pt",map_location=torch.device('cpu'))
    model.eval()
    tmp,z=model.sample_mesh()
    tmp=transformer.inverse_transform(tmp)
    latent_space=torch.zeros(NUMBER_SAMPLES,np.prod(z.shape))
    error=0
    for i in trange(NUMBER_SAMPLES):
        tmp,z=model.sample_mesh()
        tmp=transformer.inverse_transform(tmp)
        latent_space[i]=z
        tmp=tmp.cpu().detach().numpy().reshape(-1,3)
        tmp=tmp.reshape(-1,3)
        error=error+np.min(np.linalg.norm(tmp-data2,axis=1))/np.linalg.norm(data2)/NUMBER_SAMPLES
        area_sampled[i]=area(tmp,triangles)
        temp[i]=tmp.reshape(-1)
        meshio.write_points_cells("./nn/inference_objects/"+name+"_{}.ply".format(i), tmp,[])
    moment_tensor_sampled=np.zeros((NUMBER_SAMPLES,3,3))

    perc_pass=torch.sum(disc.predict([torch.tensor(temp.astype(np.float32)).reshape(NUMBER_SAMPLES,-1),_]))/NUMBER_SAMPLES
    print("Percentage of passing samples of ", name, " is ", perc_pass.detach().numpy())
    print("Variance of ",name," is", np.sum(np.var(temp.reshape(NUMBER_SAMPLES,-1),axis=0)))
    np.save("nn/inference_objects/"+name+"_latent.npy",latent_space.detach().numpy())
    
    for j in range(3):
        for k in range(3):
            moment_tensor_sampled[:,j,k]=np.mean(temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)
    variance=np.sum(np.var(temp,axis=0))
    variance_data=np.sum(np.var(data2.reshape(NUMBER_SAMPLES,-1),axis=0))
    mmd_data=relmmd(disc.compute_latent([torch.tensor(temp.astype(np.float32)).reshape(NUMBER_SAMPLES,-1),_]),disc.compute_latent([torch.tensor(data2.astype(np.float32)).reshape(NUMBER_SAMPLES,-1),_]))
    np.save("nn/geometrical_measures/moment_tensor_data.npy",moment_tensor_data)
    np.save("nn/geometrical_measures/moment_tensor_"+name+".npy",moment_tensor_sampled)
    print("Saved moments")
    np.save("nn/geometrical_measures/area_data.npy",area_data)
    np.save("nn/geometrical_measures/area_"+name+".npy",area_sampled)
    print("Saved Areas")
    np.save("nn/geometrical_measures/variance_"+name+".npy",variance)
    np.save("nn/geometrical_measures/variance_data.npy",variance_data)
    print("Saved variances")
    np.save("nn/geometrical_measures/perc_pass_"+name+".npy",perc_pass)
    np.save("nn/geometrical_measures/mmd_"+name+".npy",mmd_data)
    np.save("nn/geometrical_measures/rel_error_"+name+".npy",error)
    print("Saved error")
    np.save("nn/inference_objects/"+name+".npy",temp.reshape(NUMBER_SAMPLES,-1,3))
