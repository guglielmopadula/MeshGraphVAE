#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
from datawrapper.data import Data
import sys
import meshio
import qllr
from models.AE import AE
from pytorch_lightning import LightningModule
import copy
from tqdm import trange
from models.losses.losses import relmmd
from torch import nn
import torch
import numpy as np
from sparselinear import BayesianSparseLinear,BayesianSparsePooler,BayesianSparseUnpooler,SparsePooler,SparseUnpooler
from pytorch_lightning import Trainer
torch.autograd.set_detect_anomaly(True)
from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(True)
class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


NUM_WORKERS = os.cpu_count()//2
use_cuda=True if torch.cuda.is_available() else False


use_cuda=True if torch.cuda.is_available() else False

use_cuda=False
AVAIL_GPUS=1 if use_cuda else 0
MAX_EPOCHS=500
REDUCED_DIMENSION=20
NUM_TRAIN_SAMPLES=400#400
NUM_TEST_SAMPLES=200#200
BATCH_SIZE = 200
LATENT_DIM=5#3
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TRAIN_SAMPLES+NUM_TEST_SAMPLES

NUM_SAMPLED=600

data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/data.npy"),
          use_cuda=use_cuda)
                               

def custom_test(model,data):
    iterator=iter(data.test_dataloader())
    n_batches=data.num_test//data.batch_size
    tot_loss=0
    for i in range(n_batches):
        batch=next(iterator)
        loss=model.test_step(batch,0)
        tot_loss=tot_loss+loss
    tot_loss=tot_loss/n_batches
    return tot_loss




index00=np.array(list(np.load("graphs/from_0_to_0.npy",allow_pickle=True).item()))

##Removing the first graph element as we are using an implicit constraint
index00=index00[index00[:,1]!=0]
index00=index00[index00[:,0]!=0]

##Reset the indexes to 0
index00=index00-1

#Same for index01


index01=torch.tensor(np.load("graphs/from_0_to_1.npy"))
index01=index01[index01[:,0]!=0]
index01[:,0]=index01[:,0]-1
index11=torch.tensor(np.array(list(np.load("graphs/from_1_to_1.npy",allow_pickle=True).item())))
index12=torch.tensor(np.load("graphs/from_1_to_2.npy"))
index22=torch.tensor(np.array(list(np.load("graphs/from_2_to_2.npy",allow_pickle=True).item())))
index23=torch.tensor(np.load("graphs/from_2_to_3.npy"))
index33=torch.tensor(np.array(list(np.load("graphs/from_3_to_3.npy",allow_pickle=True).item())))
index34=torch.tensor(np.load("graphs/from_3_to_4.npy"))
index44=torch.tensor(np.array(list(np.load("graphs/from_4_to_4.npy",allow_pickle=True).item())))
index45=torch.tensor(np.load("graphs/from_4_to_5.npy"))
index55=torch.tensor(np.array(list(np.load("graphs/from_5_to_5.npy",allow_pickle=True).item())))

#transpose
index00=index00.T
index01=index01.T
index11=index11.T
index12=index12.T
index22=index22.T
index23=index23.T
index33=index33.T
index34=index34.T
index44=index44.T
index45=index45.T
index55=index55.T

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

gatto=getinfo("data/bunny_{}.ply".format(0))
bar=np.mean(gatto,axis=0)
print(bar)
num_points=len(gatto)
print(num_points)

def matrix(x):
    A=np.tile(np.eye(3)/num_points,num_points).reshape(1,3,3*num_points)
    A=np.tile(A,(x.shape[0],1,1))
    return A
lin_indices=[i for i in range(3*num_points)]

transformer=qllr.QLLR(matrix,bar,lin_indices,[])

a=transformer.transform(gatto.reshape(1,-1))
b=transformer.inverse_transform(a.reshape(1,-1))


class BayesianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slin1=BayesianSparsePooler(3,1,index01,-1000)
        self.slin2=BayesianSparsePooler(1,1,index12,-1000)
        self.slin3=BayesianSparsePooler(1,1,index23,-1000)
        self.slin4=BayesianSparsePooler(1,1,index34,-1000)
        self.slin5=BayesianSparsePooler(1,1,index45,-1000)
        self.relu=nn.ReLU()
        self.batch1=nn.BatchNorm1d(self.slin1.size2)
        self.batch2=nn.BatchNorm1d(self.slin2.size2)
        self.batch3=nn.BatchNorm1d(self.slin3.size2)
        self.batch4=nn.BatchNorm1d(self.slin4.size2)
        self.batch5=nn.BatchNorm1d(self.slin5.size2,affine=False,track_running_stats=False)
        self.dropout=nn.Dropout(DROP_PROB)

    def forward(self,x):
        kl_tot=0
        x=x.reshape(BATCH_SIZE,-1)
        x=x.unsqueeze(-1)
        x,kl=self.slin1(x)
        kl_tot=kl_tot+kl
        x=self.batch1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin2(x)
        kl_tot=kl_tot+kl
        x=self.batch2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin3(x)
        kl_tot=kl_tot+kl
        x=self.batch3(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin4(x)
        kl_tot=kl_tot+kl
        x=self.batch4(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin5(x)
        kl_tot=kl_tot+kl
        x=x.squeeze(-1)
        return x,kl_tot

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slin1=SparsePooler(3,1,index01)
        self.slin2=SparsePooler(1,1,index12)
        self.slin3=SparsePooler(1,1,index23)
        self.slin4=SparsePooler(1,1,index34)
        self.slin5=SparsePooler(1,1,index45)
        self.relu=nn.ReLU()
        self.batch1=nn.BatchNorm1d(self.slin1.size2)
        self.batch2=nn.BatchNorm1d(self.slin2.size2)
        self.batch3=nn.BatchNorm1d(self.slin3.size2)
        self.batch4=nn.BatchNorm1d(self.slin4.size2)
        self.batch5=nn.BatchNorm1d(self.slin5.size2,affine=False,track_running_stats=False)
        self.dropout=nn.Dropout(DROP_PROB)

    def forward(self,x):
        kl_tot=0
        x=x.reshape(BATCH_SIZE,-1)
        x=x.unsqueeze(-1)
        x=self.slin1(x)
        x=self.batch1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin2(x)
        x=self.batch2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin3(x)
        x=self.batch3(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin4(x)
        x=self.batch4(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin5(x)
        x=x.squeeze(-1)
        return x




class BayesianDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slin1=BayesianSparseUnpooler(1,1,index45,-1000)
        self.slin2=BayesianSparseUnpooler(1,1,index34,-1000)
        self.slin3=BayesianSparseUnpooler(1,1,index23,-1000)
        self.slin4=BayesianSparseUnpooler(1,1,index12,-1000)
        self.slin5=BayesianSparseUnpooler(1,3,index01,-1000)
        self.relu=nn.ReLU()
        self.batch1=nn.BatchNorm1d(self.slin1.size2)
        self.batch2=nn.BatchNorm1d(self.slin2.size2)
        self.batch3=nn.BatchNorm1d(self.slin3.size2)
        self.batch4=nn.BatchNorm1d(self.slin4.size2)
        self.batch5=nn.BatchNorm1d(self.slin5.size2)
        self.dropout=nn.Dropout(DROP_PROB)
    def forward(self,x):
        kl_tot=0
        x=x.unsqueeze(-1)
        x,kl=self.slin1(x)
        kl_tot=kl_tot+kl
        x=self.batch1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin2(x)
        kl_tot=kl_tot+kl
        x=self.batch2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin3(x)
        kl_tot=kl_tot+kl
        x=self.batch3(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin4(x)
        kl_tot=kl_tot+kl
        x=self.batch4(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.slin5(x)
        kl_tot=kl_tot+kl
        x=x.squeeze(-1)
        return x,kl_tot


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slin1=SparseUnpooler(1,1,index45)
        self.slin2=SparseUnpooler(1,1,index34)
        self.slin3=SparseUnpooler(1,1,index23)
        self.slin4=SparseUnpooler(1,1,index12)
        self.slin5=SparseUnpooler(1,3,index01)
        self.relu=nn.ReLU()
        self.batch1=nn.BatchNorm1d(self.slin1.size2)
        self.batch2=nn.BatchNorm1d(self.slin2.size2)
        self.batch3=nn.BatchNorm1d(self.slin3.size2)
        self.batch4=nn.BatchNorm1d(self.slin4.size2)
        self.batch5=nn.BatchNorm1d(self.slin5.size2)
        self.dropout=nn.Dropout(DROP_PROB)
    def forward(self,x):
        x=x.unsqueeze(-1)
        x=self.slin1(x)
        x=self.batch1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin2(x)
        x=self.batch2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin3(x)
        x=self.batch3(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin4(x)
        x=self.batch4(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.slin5(x)
        x=x.squeeze(-1)
        return x

def area(vertices, triangles):
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1) / 2
    return np.sum(a)


class BGAE(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = BayesianEncoder()
        self.decoder = BayesianDecoder()
        self.automatic_optimization=False
        self.train_losses=[]
        self.eval_losses=[]


    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        x=batch
        z,kl1=self.encoder(x)
        x_hat,kl2=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x_hat-x)+0*(kl1+kl2)
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()
        self.train_losses.append(loss.item())
        print(loss/torch.linalg.norm(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x=batch
        z,kl1=self.encoder(x)
        x_hat,kl2=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.eval_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        z,kl1=self.encoder(x)
        x_hat,kl2=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-1)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.slin1.weight_mean.device
        latent_size=self.decoder.slin1.size1
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,latent_size)

        if var==None:
            var=torch.ones(1,latent_size)

        z = torch.sqrt(var)*torch.randn(1,latent_size)+mean
        z=z.to(device)
        temp_interior,_=self.decoder(z)
        return temp_interior,z
     

class GAE(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.automatic_optimization=False
        self.train_losses=[]
        self.eval_losses=[]


    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x_hat-x)
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()
        self.train_losses.append(loss.item())
        print(loss/torch.linalg.norm(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.eval_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-1)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.slin1.values.device
        latent_size=self.decoder.slin1.size1
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,latent_size)

        if var==None:
            var=torch.ones(1,latent_size)

        z = torch.sqrt(var)*torch.randn(1,latent_size)+mean
        z=z.to(device)
        temp_interior=self.decoder(z)
        return temp_interior,z



torch.manual_seed(100)
np.random.seed(100)
if data.use_cuda:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,log_every_n_steps=1,
                            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                            )
else:
    trainer=Trainer(max_epochs=MAX_EPOCHS,log_every_n_steps=1,
                        plugins=[DisabledSLURMEnvironment(auto_requeue=False)],accelerator="cpu"
                        )

model=GAE()

trainer.fit(model, data)


torch.save(model,"./nn/saved_models/GAE.pt")


np.save("./nn/saved_models/BGAE_train_loss.npy",np.array(model.train_losses))
np.save("./nn/saved_models/BGAE__eval_loss.npy",np.array(model.eval_losses))

model=model.eval()

with torch.no_grad():
    print(custom_test(model,data))


d={
  GAE: "GAE", 
  }

data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/data.npy"),
          use_cuda=False)

triangles=np.load("data/triangles.npy")
data=data.data[:].cpu().numpy().reshape(NUMBER_SAMPLES,-1)
moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))
area_data=np.zeros(NUMBER_SAMPLES)
_=0
data2=data.copy()
print(data2.reshape(NUMBER_SAMPLES,-1).shape)
data2=transformer.inverse_transform(data2.reshape(NUMBER_SAMPLES,-1))
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
    tmp=tmp.cpu().detach().numpy()
    tmp=transformer.inverse_transform(tmp)
    temp=np.zeros((NUM_SAMPLED,tmp.reshape(-1).shape[0]))
    latent_space=torch.zeros(NUM_SAMPLED,np.prod(z.shape))
    error=0
    for i in trange(NUM_SAMPLED):
        tmp,z=model.sample_mesh()
        tmp=tmp.cpu().detach().numpy()
        tmp=transformer.inverse_transform(tmp)
        latent_space[i]=z
        tmp=tmp.reshape(-1,3)
        error=error+np.min(np.linalg.norm(tmp-data2,axis=1))/np.linalg.norm(data2)/NUM_SAMPLED
        area_sampled[i]=area(tmp,triangles)
        temp[i]=tmp.reshape(-1)
        meshio.write_points_cells("./nn/inference_objects/"+name+"_{}.ply".format(i), tmp,[])
    moment_tensor_sampled=np.zeros((NUM_SAMPLED,3,3))

    print("Variance of ",name," is", np.sum(np.var(temp.reshape(NUM_SAMPLED,-1),axis=0)))
    np.save("nn/inference_objects/"+name+"_latent.npy",latent_space.detach().numpy())
    
    for j in range(3):
        for k in range(3):
            moment_tensor_sampled[:,j,k]=np.mean(temp.reshape(NUM_SAMPLED,-1,3)[:,:,j]*temp.reshape(NUM_SAMPLED,3)[:,:,k],axis=1)
    variance=np.sum(np.var(temp,axis=0))
    variance_data=np.sum(np.var(data2.reshape(NUM_SAMPLED,-1),axis=0))
    np.save("nn/geometrical_measures/moment_tensor_data.npy",moment_tensor_data)
    np.save("nn/geometrical_measures/moment_tensor_"+name+".npy",moment_tensor_sampled)
    print("Saved moments")
    np.save("nn/geometrical_measures/area_data.npy",area_data)
    np.save("nn/geometrical_measures/area_"+name+".npy",area_sampled)
    print("Saved Areas")
    np.save("nn/geometrical_measures/variance_"+name+".npy",variance)
    np.save("nn/geometrical_measures/variance_data.npy",variance_data)
    print("Saved variances")
    np.save("nn/geometrical_measures/rel_error_"+name+".npy",error)
    print("Saved error")
    np.save("nn/inference_objects/"+name+".npy",temp.reshape(NUM_SAMPLED,-1,3))
