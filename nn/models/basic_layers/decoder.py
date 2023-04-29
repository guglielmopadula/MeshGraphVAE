from models.basic_layers.lbr import LBR
from models.basic_layers.doublec import Double
from models.basic_layers.ctu import CTU
from torch_geometric.nn import ChebConv

import numpy as np
from torch import nn
import torch

class Decoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,batch_size,drop_prob,barycenter,volume,triangles,l,adj,points_triangles_matrix):
        super().__init__()
        self.data_shape=data_shape
        self.points_triangles_matrix=points_triangles_matrix
        self.batch_size=batch_size
        self.barycenter=barycenter
        self.volume=volume
        self.triangles=triangles
        self.l=l
        self.adj=adj
        self.drop_prob=drop_prob
        self.fc_interior_1 = nn.Linear(latent_dim, (1+torch.amax(adj[3]))*hidden_dim)
        self.fc_interior_2 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_3 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_4 = LBR((1+torch.amax(adj[3]))*hidden_dim, (1+torch.amax(adj[3]))*hidden_dim,drop_prob)
        self.fc_interior_5 = ChebConv(hidden_dim, hidden_dim,3)
        self.fc_interior_6 = CTU(hidden_dim, hidden_dim,l[2],adj[3],adj[2],drop_prob)
        self.fc_interior_7 = CTU(hidden_dim, hidden_dim,l[1],adj[2],adj[1],drop_prob)
        self.fc_interior_8 = CTU(hidden_dim, hidden_dim,l[0],adj[1],adj[0],drop_prob)
        self.fc_interior_9 = ChebConv(hidden_dim, 3,3)
        self.doublec=Double(batch_size=self.batch_size,barycenter=self.barycenter,volume=self.volume,triangles=self.triangles,points_triangles_matrix=points_triangles_matrix)
    
    def forward(self, z):
        z=self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z))))
        z=z.reshape(z.shape[0],1+torch.amax(self.adj[3]),-1)
        z=self.fc_interior_5(z,self.adj[3])
        z=self.fc_interior_8(self.fc_interior_7(self.fc_interior_6(z)))
        z=self.fc_interior_9(z,self.adj[0])
        z=self.doublec(z)
        return z
 
