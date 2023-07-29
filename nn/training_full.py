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
from pytorch_lightning import LightningModule
import copy
from tqdm import trange
from models.losses.losses import relmmd
from torch import nn
import torch
import numpy as np
from sparselinear import BayesianSparseLinear,BayesianSparsePooler,BayesianSparseUnpooler,SparsePooler,SparseUnpooler,SparseLinear
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

NUM_SAMPLED=600#NUMBER_SAMPLES

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


def reconstuct(model,data):
    iterator=iter(data.test_dataloader())
    n_batches=data.num_test//data.batch_size
    tot_loss=0
    for i in range(n_batches):
        batch=next(iterator)
        batch=model.decoder(model.encoder(batch))
        for j in trange(len(batch)):
            meshio.write_points_cells("./nn/inference_objects/GAE_recon_{}.ply".format(i*200+j), batch[j].detach().numpy().reshape(-1,3),[])
    return tot_loss


index00=torch.tensor(np.array(list(np.load("graphs/from_0_to_0.npy",allow_pickle=True).item())))
index01=torch.tensor(np.load("graphs/from_0_to_1.npy"))
index11=torch.tensor(np.array(list(np.load("graphs/from_1_to_1.npy",allow_pickle=True).item())))
index12=torch.tensor(np.load("graphs/from_1_to_2.npy"))
index22=torch.tensor(np.array(list(np.load("graphs/from_2_to_2.npy",allow_pickle=True).item())))
index23=torch.tensor(np.load("graphs/from_2_to_3.npy"))
index33=torch.tensor(np.array(list(np.load("graphs/from_3_to_3.npy",allow_pickle=True).item())))
index34=torch.tensor(np.load("graphs/from_3_to_4.npy"))
index44=torch.tensor(np.array(list(np.load("graphs/from_4_to_4.npy",allow_pickle=True).item())))
index45=torch.tensor(np.load("graphs/from_4_to_5.npy"))
index55=torch.tensor(np.array(list(np.load("graphs/from_5_to_5.npy",allow_pickle=True).item())))
index56=torch.tensor(np.load("graphs/from_5_to_6.npy"))
index66=torch.tensor(np.array(list(np.load("graphs/from_6_to_6.npy",allow_pickle=True).item())))

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
index56=index56.T
index66=index66.T

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

class BayesianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spool1=BayesianSparsePooler(3,3,index01,-100)
        self.spool2=BayesianSparsePooler(3,3,index12,-100)
        self.spool3=BayesianSparsePooler(3,3,index23,-100)
        self.spool4=BayesianSparsePooler(3,3,index34,-100)
        self.spool5=BayesianSparsePooler(3,3,index45,-100)
        self.spool6=BayesianSparsePooler(3,3,index56,-100)
        self.slin0=BayesianSparseLinear(3,3,index00,-100)
        self.slin1=BayesianSparseLinear(3,3,index11,-100)
        self.slin2=BayesianSparseLinear(3,3,index22,-100)
        self.slin3=BayesianSparseLinear(3,3,index33,-100)
        self.slin4=BayesianSparseLinear(3,3,index44,-100)
        self.slin5=BayesianSparseLinear(3,3,index55,-100)
        self.slin6=BayesianSparseLinear(3,3,index66,-100)
        self.relu=nn.ReLU()
        self.batch0l=nn.BatchNorm1d(self.slin0.size2)
        self.batch1l=nn.BatchNorm1d(self.slin1.size2)
        self.batch2l=nn.BatchNorm1d(self.slin2.size2)
        self.batch3l=nn.BatchNorm1d(self.slin3.size2)
        self.batch4l=nn.BatchNorm1d(self.slin4.size2)
        self.batch5l=nn.BatchNorm1d(self.slin5.size2)
        self.batch6l=nn.BatchNorm1d(self.slin6.size2,affine=False,track_running_stats=False)
        self.batch1p=nn.BatchNorm1d(self.spool1.size2)
        self.batch2p=nn.BatchNorm1d(self.spool2.size2)
        self.batch3p=nn.BatchNorm1d(self.spool3.size2)
        self.batch4p=nn.BatchNorm1d(self.spool4.size2)
        self.batch5p=nn.BatchNorm1d(self.spool5.size2)
        self.batch6p=nn.BatchNorm1d(self.spool6.size2)
        self.dropout=nn.Dropout(DROP_PROB)

    def forward(self,x):
        kl_tot=0
        x=x.reshape(BATCH_SIZE,-1)
        x=x.unsqueeze(-1)
        x,kl=self.slin0(x)
        kl_tot=kl+kl_tot
        x=self.batch0l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool1(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin1(x)
        kl_tot=kl+kl_tot
        x=self.batch1l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool2(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin2(x)
        kl_tot=kl+kl_tot
        x=self.batch2l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool3(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin3(x)
        kl_tot=kl+kl_tot
        x=self.batch3l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool4(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin4(x)
        kl_tot=kl+kl_tot
        x=self.batch4l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool5(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin5(x)
        kl_tot=kl+kl_tot
        x=self.batch5l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.spool6(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin6(x)
        kl_tot=kl+kl_tot
        x=x.squeeze(-1)
        x=self.batch6l(x)
        return x,kl_tot


class BayesianDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sunpool1=BayesianSparseUnpooler(3,3,index56,-100)
        self.sunpool2=BayesianSparseUnpooler(3,3,index45,-100)
        self.sunpool3=BayesianSparseUnpooler(3,3,index34,-100)
        self.sunpool4=BayesianSparseUnpooler(3,3,index23,-100)
        self.sunpool5=BayesianSparseUnpooler(3,3,index12,-100)
        self.sunpool6=BayesianSparseUnpooler(3,3,index01,-100)
        self.relu=nn.ReLU()
        self.slin6=BayesianSparseLinear(3,3,index00,-100)
        self.slin5=BayesianSparseLinear(3,3,index11,-100)
        self.slin4=BayesianSparseLinear(3,3,index22,-100)
        self.slin3=BayesianSparseLinear(3,3,index33,-100)
        self.slin2=BayesianSparseLinear(3,3,index44,-100)
        self.slin1=BayesianSparseLinear(3,3,index55,-100)
        self.slin0=BayesianSparseLinear(3,3,index66,-100)
        self.batch0l=nn.BatchNorm1d(self.slin0.size2)
        self.batch1l=nn.BatchNorm1d(self.slin1.size2)
        self.batch2l=nn.BatchNorm1d(self.slin2.size2)
        self.batch3l=nn.BatchNorm1d(self.slin3.size2)
        self.batch4l=nn.BatchNorm1d(self.slin4.size2)
        self.batch5l=nn.BatchNorm1d(self.slin5.size2)
        self.batch6l=nn.BatchNorm1d(self.slin5.size2)
        self.dropout=nn.Dropout(DROP_PROB)
    def forward(self,x):
        x=x.unsqueeze(-1)
        kl_tot=0
        x,kl=self.slin0(x)
        kl_tot=kl+kl_tot
        x=self.batch0l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.sunpool1(x)       
        kl_tot=kl+kl_tot
        x,kl=self.slin1(x)
        kl_tot=kl+kl_tot
        x=self.batch1l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.sunpool2(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin2(x)
        kl_tot=kl+kl_tot
        x=self.batch2l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.sunpool3(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin3(x)
        kl_tot=kl+kl_tot
        x=self.batch3l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.sunpool4(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin4(x)
        kl_tot=kl+kl_tot
        x=self.batch4l(x)
        x=self.relu(x)
        x=self.dropout(x)
        x,kl=self.sunpool5(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin5(x)
        kl_tot=kl+kl_tot
        x=self.batch5l(x)
        x=self.relu(x)
        x,kl=self.sunpool6(x)
        kl_tot=kl+kl_tot
        x,kl=self.slin6(x)
        kl_tot=kl+kl_tot
        x=x.squeeze(-1)
        s=x.shape
        x=x.reshape(x.shape[0],-1,3)
        x=x-torch.mean(x,axis=1).unsqueeze(1).repeat(1,x.shape[1],1)
        x=x.reshape(s)
        return x,kl





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
        loss = torch.linalg.norm(x_hat-x)+(kl1+kl2)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.slin1.weight_mean.device
        latent_size=self.decoder.slin0.size1
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,latent_size)

        if var==None:
            var=torch.ones(1,latent_size)

        z = torch.sqrt(var)*torch.randn(1,latent_size)+mean
        z=z.to(device)
        temp_interior,_=self.decoder(z)
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

model=BGAE()

trainer.fit(model, data)
 

model=model.eval()
with torch.no_grad():
    print(custom_test(model,data))


d={
  BGAE: "BGAE", 
  }

data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/data.npy"),
          use_cuda=False)
#reconstuct(model,data)

triangles=np.load("data/triangles.npy")
data=data.data[:].cpu().numpy().reshape(NUMBER_SAMPLES,-1)
moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))
area_data=np.zeros(NUMBER_SAMPLES)
_=0
data2=data.copy()
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
    tmp,z=model.sample_mesh()
    tmp=tmp.cpu().detach().numpy()
    temp=np.zeros((NUM_SAMPLED,tmp.reshape(-1).shape[0]))
    latent_space=torch.zeros(NUM_SAMPLED,np.prod(z.shape))
    error=0
    for i in trange(NUM_SAMPLED):
        tmp,z=model.sample_mesh()
        tmp=tmp.cpu().detach().numpy()
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
            moment_tensor_sampled[:,j,k]=np.mean(temp.reshape(NUM_SAMPLED,-1,3)[:,:,j]*temp.reshape(NUM_SAMPLED,-1,3)[:,:,k],axis=1)
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


area_data=np.load("nn/geometrical_measures/area_data.npy")
area_sampled=np.load("nn/geometrical_measures/area_BGAE.npy")
import matplotlib.pyplot as plt
fig2,ax2=plt.subplots()
ax2.set_title("Area")
_=ax2.hist(area_data,8,label='real',histtype='step',linestyle='solid',density=True)
_=ax2.hist(area_sampled,8,label='sampled',histtype='step',linestyle='dotted',density=True)
ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("test.pdf")