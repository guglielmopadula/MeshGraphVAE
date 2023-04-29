from models.basic_layers.lbr import LBR
from models.basic_layers.ctp import CTP
from torch_geometric.nn import ChebConv
import numpy as np
import torch
from torch import nn
class Encoder_base(nn.Module):
    def __init__(self, latent_dim,hidden_dim,data_shape,drop_prob,batch_size,l,adj):
        super().__init__()
        self.data_shape=data_shape
        self.batch_size=batch_size
        self.latent_dim=latent_dim
        self.l=l
        self.adj=adj    
        self.drop_prob=drop_prob
        self.fc_interior_1 = ChebConv(3, hidden_dim,3)
        self.fc_interior_2 = CTP(hidden_dim, hidden_dim,l[0],adj[1],drop_prob)
        self.fc_interior_3 = CTP(hidden_dim, hidden_dim,l[1],adj[2],drop_prob)
        self.fc_interior_4 = CTP(hidden_dim, hidden_dim,l[2],adj[3],drop_prob)
        self.fc_interior_5 = ChebConv(hidden_dim, hidden_dim,3)
        self.fc_interior_6 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_7 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_8 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_9 = nn.Linear((1+torch.amax(adj[3]))*hidden_dim, latent_dim)

        self.tanh=nn.Tanh()
        self.batch_mu_1=nn.BatchNorm1d(self.latent_dim,affine=False,track_running_stats=False)


    def forward(self, x):
        x=self.fc_interior_1(x,self.adj[0])
        x=self.fc_interior_2(x)
        x=self.fc_interior_3(x)
        x=self.fc_interior_4(x)
        x=self.fc_interior_5(x,self.adj[3])
        x=x.reshape(x.shape[0],-1)
        mu_1=self.fc_interior_9(self.fc_interior_8(self.fc_interior_7(self.fc_interior_6(x))))
        mu_1=self.batch_mu_1(mu_1)
        return mu_1
 
