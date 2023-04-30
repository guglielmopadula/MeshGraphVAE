
import numpy as np
import meshio
import torch
import tetgen
from scipy.sparse import coo_array
from pprint import pprint




def volume_tet_m(points,elem):
    points=points[:,elem,:]
    print(points.shape)
    points=np.concatenate((points,np.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    print(points.shape)
    points=np.transpose(points,axes=(0,1,3,2))
    print(points.shape)
    return np.sum(np.linalg.det(points),axis=1)/6

def get_coeff_x(points,elem,arr):
    n_samples=len(points)
    indices=np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    x_index=0
    points=points[:,elem,:]
    points=np.concatenate((points,np.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=np.transpose(points,axes=(0,1,3,2))
    points=points[:,:,1:,indices]
    points=np.transpose(points,axes=(0,1,3,2,4))
    a=np.linalg.det(points)*(-1)**(x_index+1+np.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr




def get_coeff_y(points,elem,arr):
    n_samples=len(points)
    indices=np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    y_index=1
    points=points[:,elem,:]
    points=np.concatenate((points,np.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=np.transpose(points,axes=(0,1,3,2))
    points=points[:,:,[0,2,3]]
    points=points[:,:,:,indices]
    points=np.transpose(points,axes=(0,1,3,2,4))
    a=np.linalg.det(points)*(-1)**(y_index+1+np.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr


def get_coeff_z(points,elem,arr):
    n_samples=len(points)
    indices=np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    z_index=2
    points=points[:,elem,:]
    points=np.concatenate((points,np.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=np.transpose(points,axes=(0,1,3,2))
    points=points[:,:,[0,1,3]]
    points=points[:,:,:,indices]
    points=np.transpose(points,axes=(0,1,3,2,4))
    a=np.linalg.det(points)*(-1)**(z_index+1+np.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr

def get_volume(points,elem):
    points=points[:,elem,:]
    points=torch.concatenate((points,torch.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=torch.transpose(points,2,3)
    return torch.sum(torch.linalg.det(points),axis=1)/6



def get_coeff_x_torch(points,elem,arr):
    n_samples=len(points)
    indices=torch.tensor([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    x_index=0
    points=points[:,elem,:]
    points=torch.concatenate((points,torch.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=torch.transpose(points,2,3)
    points=points[:,:,1:] 
    points=points[:,:,:,indices]
    points=torch.transpose(points,2,3)
    a=torch.linalg.det(points)*(-1)**(x_index+1+torch.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr

def get_coeff_y_torch(points,elem,arr):
    n_samples=len(points)
    indices=torch.tensor([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    y_index=1
    points=points[:,elem,:]
    points=torch.concatenate((points,torch.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=torch.transpose(points,2,3)
    points=points[:,:,[0,2,3]] 
    points=points[:,:,:,indices]
    points=torch.transpose(points,2,3)
    a=torch.linalg.det(points)*(-1)**(y_index+1+torch.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr


def get_coeff_z_torch(points,elem,arr):
    n_samples=len(points)
    indices=torch.tensor([[1,2,3],[0,2,3],[0,1,3],[0,1,2]],dtype=int)
    z_index=2
    points=points[:,elem,:]
    points=torch.concatenate((points,torch.ones((points.shape[0],points.shape[1],4,1))),axis=3)
    points=torch.transpose(points,2,3)
    points=points[:,:,[0,1,3]] 
    points=points[:,:,:,indices]
    points=torch.transpose(points,2,3)
    a=torch.linalg.det(points)*(-1)**(z_index+1+torch.arange(4)+1)/6
    return a.reshape(n_samples,-1)@arr



def get_point_triangle_matrix(points,elem):
    n_points=len(points)
    n_tetras=len(elem)
    l=[]
    for i in range(n_tetras):
        for j in range(4):
            l.append([i,j,elem[i][j]])
    l=np.array(l)
    l=np.concatenate(((l[:,0]*4+l[:,1]).reshape(-1,1),l[:,2].reshape(-1,1)),axis=1)
    row=l.T[0]
    col=l.T[1]
    data=np.ones(len(col),dtype=np.float32)
    arr=coo_array((data, (row, col)), shape=(4*n_tetras, n_points))
    return arr


def scipy_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def get_coeff_x_slow(points,triangles):
    p=np.zeros(points.shape[0])
    #mesh=points_triangle_indexing(points,triangles)
    mesh=points[triangles]
    M=np.zeros((triangles.shape[0],triangles.shape[1],4))
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,1,2]*M[i,2,1]+M[i,1,3]*M[i,2,1]+M[i,1,1]*M[i,2,2]-M[i,1,3]*M[i,2,2]-M[i,1,1]*M[i,2,3]+M[i,1,2]*M[i,2,3])*(1)/6
        p[k[1]]=p[k[1]]+(-M[i,1,2]*M[i,2,0]+M[i,1,3]*M[i,2,0]+M[i,1,0]*M[i,2,2]-M[i,1,3]*M[i,2,2]-M[i,1,0]*M[i,2,3]+M[i,1,2]*M[i,2,3])*(-1)/6
        p[k[2]]=p[k[2]]+(-M[i,1,1]*M[i,2,0]+M[i,1,3]*M[i,2,0]+M[i,1,0]*M[i,2,1]-M[i,1,3]*M[i,2,1]-M[i,1,0]*M[i,2,3]+M[i,1,1]*M[i,2,3])*(1)/6
        p[k[3]]=p[k[3]]+(-M[i,1,1]*M[i,2,0]+M[i,1,2]*M[i,2,0]+M[i,1,0]*M[i,2,1]-M[i,1,2]*M[i,2,1]-M[i,1,0]*M[i,2,2]+M[i,1,1]*M[i,2,2])*(-1)/6
    return p


def get_coeff_y_slow(points,triangles):
    p=np.zeros(points.shape[0])
    #mesh=points_triangle_indexing(points,triangles)
    mesh=points[triangles]
    M=np.zeros((triangles.shape[0],triangles.shape[1],4))
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,0,2]*M[i,2,1]+M[i,0,3]*M[i,2,1]+M[i,0,1]*M[i,2,2]-M[i,0,3]*M[i,2,2]-M[i,0,1]*M[i,2,3]+M[i,0,2]*M[i,2,3])*(-1)/6
        p[k[1]]=p[k[1]]+(-M[i,0,2]*M[i,2,0]+M[i,0,3]*M[i,2,0]+M[i,0,0]*M[i,2,2]-M[i,0,3]*M[i,2,2]-M[i,0,0]*M[i,2,3]+M[i,0,2]*M[i,2,3])*(1)/6
        p[k[2]]=p[k[2]]+(-M[i,0,1]*M[i,2,0]+M[i,0,3]*M[i,2,0]+M[i,0,0]*M[i,2,1]-M[i,0,3]*M[i,2,1]-M[i,0,0]*M[i,2,3]+M[i,0,1]*M[i,2,3])*(-1)/6
        p[k[3]]=p[k[3]]+(-M[i,0,1]*M[i,2,0]+M[i,0,2]*M[i,2,0]+M[i,0,0]*M[i,2,1]-M[i,0,2]*M[i,2,1]-M[i,0,0]*M[i,2,2]+M[i,0,1]*M[i,2,2])*(1)/6
    return p


def get_coeff_z_slow(points,triangles):
    p=np.zeros(points.shape[0])
    #mesh=points_triangle_indexing(points,triangles)
    mesh=points[triangles]
    M=np.zeros((triangles.shape[0],triangles.shape[1],4))
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,0,2]*M[i,1,1]+M[i,0,3]*M[i,1,1]+M[i,0,1]*M[i,1,2]-M[i,0,3]*M[i,1,2]-M[i,0,1]*M[i,1,3]+M[i,0,2]*M[i,1,3])*(1)/6
        p[k[1]]=p[k[1]]+(-M[i,0,2]*M[i,1,0]+M[i,0,3]*M[i,1,0]+M[i,0,0]*M[i,1,2]-M[i,0,3]*M[i,1,2]-M[i,0,0]*M[i,1,3]+M[i,0,2]*M[i,1,3])*(-1)/6
        p[k[2]]=p[k[2]]+(-M[i,0,1]*M[i,1,0]+M[i,0,3]*M[i,1,0]+M[i,0,0]*M[i,1,1]-M[i,0,3]*M[i,1,1]-M[i,0,0]*M[i,1,3]+M[i,0,1]*M[i,1,3])*(1)/6
        p[k[3]]=p[k[3]]+(-M[i,0,1]*M[i,1,0]+M[i,0,2]*M[i,1,0]+M[i,0,0]*M[i,1,1]-M[i,0,2]*M[i,1,1]-M[i,0,0]*M[i,1,2]+M[i,0,1]*M[i,1,2])*(-1)/6
    return p


points=meshio.read("data/Stanford_Bunny_red.ply").points
elem=np.load("data/tetras.npy")
points=points.reshape(1,-1,3)
point_r=np.repeat(points,6,axis=0)
points=points.reshape(-1,3)
point_triangles_matrix=get_point_triangle_matrix(points,elem)
a=get_coeff_x(point_r,elem,point_triangles_matrix)
a_2=get_coeff_x_slow(points,elem)
b_2=get_coeff_y_slow(points,elem)
c_2=get_coeff_z_slow(points,elem)

print(a_2@points[:,0])
print(b_2@points[:,1])
print(c_2@points[:,2])

print(a@points[:,0])
b=get_coeff_y(point_r,elem,point_triangles_matrix)
print(b@points[:,1])
c=get_coeff_z(point_r,elem,point_triangles_matrix)
print(c@points[:,2])

points=torch.tensor(points,dtype=torch.float)
point_r=torch.tensor(point_r,dtype=torch.float)
point_triangles_matrix=scipy_to_torch(point_triangles_matrix)
elem=torch.tensor(elem,dtype=int)
print(volume_tet_m_torch(point_r,elem))
a=get_coeff_x_torch(point_r,elem,point_triangles_matrix)
print(a@points[:,0])
b=get_coeff_y_torch(point_r,elem,point_triangles_matrix)
print(b@points[:,1])
c=get_coeff_z_torch(point_r,elem,point_triangles_matrix)
print(c@points[:,2])

