import meshio
import numpy as np
from cpffd import *
a=meshio.read("data/Stanford_Bunny.stl")
from tqdm import trange
p=a.points.astype(float)
np.random.seed(0)
triangles=a.cells_dict["triangle"]
def scale_normalize(points):
    minim=np.min(points,axis=0)
    points=points-minim
    scale=np.max(points)
    points=points/scale
    return points,minim,scale

def restore(points,scale,minim):
    return points*scale+minim

p,minim,scale=scale_normalize(p)
print(np.min(p))
print(np.max(p))
M=np.eye(27)
n_x=3
n_y=3
n_z=3
mask=np.ones((n_x,n_y,n_z),dtype=int)
mask[:,:,0]=0


vpffd=cpffd.BPFFD((n_x,n_y,n_z))
a=0.05
M=np.eye(np.sum(mask)*3)
indices_c=np.arange(n_x*n_y*n_z)[mask.reshape(-1).astype(bool)]
indices_c.sort()


for i in trange(7):
    vpffd.array_mu_x=a*np.random.rand(*vpffd.array_mu_x.shape)*np.arange(n_z).reshape(1,1,-1)
    vpffd.array_mu_y=a*np.random.rand(*vpffd.array_mu_y.shape)*np.arange(n_z).reshape(1,1,-1)
    vpffd.array_mu_z=a*np.random.rand(*vpffd.array_mu_z.shape)*np.arange(n_z).reshape(1,1,-1)
    pdef=vpffd.barycenter_ffd_adv(p,M,indices_c)
    pdef=restore(pdef,scale,minim)
    meshio.write_points_cells("data/Barycenter_data/bunny_{}.stl".format(i),pdef,{"triangle":triangles})


