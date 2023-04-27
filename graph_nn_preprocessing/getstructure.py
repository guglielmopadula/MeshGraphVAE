import numpy as np
import meshio
from tqdm import trange

orig_mesh=meshio.read("data/Stanford_Bunny.stl")
mesh_red_1=meshio.read("data/mesh_red_1.stl")
mesh_red_2=meshio.read("data/mesh_red_2.stl")
mesh_red_3=meshio.read("data/mesh_red_3.stl")
mesh_red_4=meshio.read("data/mesh_red_4.stl")

def get_simplified_list(mesh_1,mesh_2):
    points1=mesh_1.points
    points2=mesh_2.points
    l=[]
    for i in trange(len(points2)):
        index=np.argmin(np.linalg.norm(points1-points2[i],axis=1))
        l.append(index)
    
    return l

def get_adjacency(matrix):
    faces=matrix.cells_dict["triangle"]
    adj=[]
    for i in trange(len(faces)):
        tmp=faces[i]
        adj.append([tmp[0],tmp[1]])
        adj.append([tmp[1],tmp[0]])
        adj.append([tmp[0],tmp[2]])
        adj.append([tmp[2],tmp[0]])
        adj.append([tmp[1],tmp[2]])
        adj.append([tmp[2],tmp[1]])
    return adj



adj0=get_adjacency(orig_mesh)
adj1=get_adjacency(mesh_red_1)
adj2=get_adjacency(mesh_red_2)
adj3=get_adjacency(mesh_red_3)
adj4=get_adjacency(mesh_red_4)
l1=get_simplified_list(orig_mesh,mesh_red_1)
l2=get_simplified_list(mesh_red_1,mesh_red_2)
l3=get_simplified_list(mesh_red_2,mesh_red_3)
l4=get_simplified_list(mesh_red_3,mesh_red_4)
l1=np.array(l1)
l2=np.array(l2)
l3=np.array(l3)
l4=np.array(l4)
np.save("nn_preprocessing/subset1.npy",l1)
np.save("nn_preprocessing/subset2.npy",l2)
np.save("nn_preprocessing/subset3.npy",l3)
np.save("nn_preprocessing/subset4.npy",l4)
np.save("nn_preprocessing/adj0.npy",adj0)
np.save("nn_preprocessing/adj1.npy",adj1)
np.save("nn_preprocessing/adj2.npy",adj2)
np.save("nn_preprocessing/adj3.npy",adj3)
np.save("nn_preprocessing/adj4.npy",adj4)

