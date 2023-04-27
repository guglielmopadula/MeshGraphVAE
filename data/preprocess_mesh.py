import numpy as np
import meshio

mesh=meshio.read("data/Stanford_Bunny.stl")
mesh.points=mesh.points-np.mean(mesh.points,axis=0)
l=np.max(mesh.points,axis=0)-np.min(mesh.points,axis=0)
mesh.points=mesh.points/np.max(l)
meshio.write("data/Stanford_Bunny_preprocessed.stl")

