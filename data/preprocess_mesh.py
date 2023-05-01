import numpy as np
import meshio
import tetgen
from scipy.sparse import coo_array
from pprint import pprint

def volume_tet(points,elem):
    points=points[elem]
    points=np.concatenate((points,np.ones((4,1))),axis=1)
    points=points.T
    return np.linalg.det(points)


mesh=meshio.read("data/Stanford_Bunny.stl")
mesh.points=mesh.points-np.min(mesh.points,axis=0)
l=np.max(mesh.points,axis=0)
mesh.points=mesh.points/np.max(l)
points=mesh.points.copy()
triangles=mesh.cells_dict["triangle"]
tgen = tetgen.TetGen(points,triangles)
nodes, elem = tgen.tetrahedralize()
print(nodes.shape)
print(np.max(elem))
#print(len(points))
s=0

points=points.reshape(1,-1,3)
for i in range(len(elem)):
    if (volume_tet(nodes,elem[i])<0):
        tmp=elem[i,0]
        elem[i,0]=elem[i,1]
        elem[i,1]=tmp


meshio.write_points_cells("data/Stanford_Bunny.ply",nodes,cells=[])
np.save("data/tetras.npy",elem)

#0.22264674936972206