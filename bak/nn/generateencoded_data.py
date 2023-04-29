#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
from datawrapper.data import Data
from datawrapper.encoded_data import EncodedData
from models.AE import AE
import torch
NUM_WORKERS = 0
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0
REDUCED_DIMENSION=30
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
NUM_VAL_SAMPLES=0
BATCH_SIZE = 200

data=torch.load("./data_objects/data.pt", map_location="cpu")
AE=torch.load("./saved_models/AE.pt",map_location=torch.device('cpu'))
encoded_data_train=AE.encoder(data.data_train[:])
encoded_data_test=AE.encoder(data.data_test[:])
torch.save(encoded_data_test,"./data_objects/encoded_data_test.pt")
torch.save(encoded_data_train,"./data_objects/encoded_data_train.pt")

encoded_data_wrapper=EncodedData(num_workers=data.num_workers,batch_size=data.batch_size,data_train=encoded_data_test,data_test=encoded_data_test,use_cuda=use_cuda,latent_dim=encoded_data_train.shape[1])
torch.save(encoded_data_wrapper,"./data_objects/encoded_data.pt")