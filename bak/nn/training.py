#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
import os
import sys
from models.AE import AE
from models.AAE import AAE
from models.VAE import VAE
from models.BEGAN import BEGAN
import copy
import torch
import numpy as np
from pytorch_lightning import Trainer

from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.set_float32_matmul_precision('high')
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
AVAIL_GPUS=1 if torch.cuda.is_available() else 0

LATENT_DIM=10
REDUCED_DIMENSION=140
NUM_TRAIN_SAMPLES=4#400
NUM_TEST_SAMPLES=3#200
BATCH_SIZE = 100
MAX_EPOCHS={"AE":500,"BEGAN":500,"AAE":500,"VAE":500}
SMOOTHING_DEGREE=1
DROP_PROB=0.1


data={"bar":Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data/Barycenter_data/rabbit_{}.stl",
          use_cuda=use_cuda,pool_vec_names=["./graph_nn_preprocessing/subset1.npy",
                                            "./graph_nn_preprocessing/subset2.npy",
                                            "./graph_nn_preprocessing/subset3.npy",
                                            "./graph_nn_preprocessing/subset4.npy"],
                                            pool_adj_names=["./graph_nn_preprocessing/adj0.npy",
                                                          "./graph_nn_preprocessing/adj1.npy",
                                                          "./graph_nn_preprocessing/adj2.npy",
                                                          "./graph_nn_preprocessing/adj3.npy",
                                                          "./graph_nn_preprocessing/adj4.npy"]
                                                          ),"vol":Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data/Volume_data/rabbit_{}.stl",
          use_cuda=use_cuda,pool_vec_names=["./graph_nn_preprocessing/subset1.npy",
                                            "./graph_nn_preprocessing/subset2.npy",
                                            "./graph_nn_preprocessing/subset3.npy",
                                            "./graph_nn_preprocessing/subset4.npy"],
                                            pool_adj_names=["./graph_nn_preprocessing/adj0.npy",
                                                          "./graph_nn_preprocessing/adj1.npy",
                                                          "./graph_nn_preprocessing/adj2.npy",
                                                          "./graph_nn_preprocessing/adj3.npy",
                                                          "./graph_nn_preprocessing/adj5.npy"]
                                                          )}



'''


d={
  AE: "AE",
  AAE: "AAE",
  VAE: "VAE", 
  BEGAN: "BEGAN",
}

if __name__ == "__main__":
    name=sys.argv[1]
    wrapper=list(d.keys())[list(d.values()).index(name)]
    torch.manual_seed(100)
    np.random.seed(100)
    if use_cuda:
        trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS[name],log_every_n_steps=1,
                                plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                )
    else:
        trainer=Trainer(max_epochs=MAX_EPOCHS[name],log_every_n_steps=1,
                            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                            )   
    model=wrapper(data_shape=data.get_reduced_size(),pca=data.pca,latent_dim=LATENT_DIM,batch_size=BATCH_SIZE,drop_prob=DROP_PROB,barycenter=data.barycenter)
    print("Training of "+name+ "has started")
    if name=="BEGAN":
        ae=torch.load("./saved_models/AE.pt",map_location="cpu")
        model.discriminator.encoder_base.load_state_dict(ae.encoder.encoder_base.state_dict())
        model.discriminator.decoder_base.load_state_dict(ae.decoder.decoder_base.state_dict())
        model.generator.decoder_base.load_state_dict(ae.decoder.decoder_base.state_dict())
    if name=="VAE":
        ae=torch.load("./saved_models/AE.pt",map_location="cpu")
        model.encoder.encoder_base.load_state_dict(ae.encoder.encoder_base.state_dict())
        model.decoder.decoder_base.load_state_dict(ae.decoder.decoder_base.state_dict())
    if name=="AAE":
        ae=torch.load("./saved_models/AE.pt",map_location="cpu")
        model.encoder.encoder_base.load_state_dict(ae.encoder.encoder_base.state_dict())
        model.decoder.decoder_base.load_state_dict(ae.decoder.decoder_base.state_dict())

    trainer.fit(model, data)
    trainer.test(model,data)
    torch.save(model,"./saved_models/"+name+".pt")


    
    

    
    
    
    
    
    
    
    
    
    
    
    '''
    
    
