import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import time
from dolfinx.fem import FunctionSpace
import pyvista
from mpi4py import MPI
from dolfinx import mesh
import tetgen
import meshio

def calculate_simulation(name,bary,write=True):
    mymesh=meshio.read(name+".stl")
    tgen = tetgen.TetGen(mymesh.points,mymesh.cells_dict["triangle"])
    nodes, elem = tgen.tetrahedralize()
    nodes=nodes-np.min(nodes,axis=0)
    gdim = 3
    shape = "tetrahedron"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    domain = mesh.create_mesh(MPI.COMM_WORLD, elem, nodes, domain)
    V = FunctionSpace(domain, ("CG", 1))
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.locate_entities_boundary(domain, dim=fdim,
                                       marker=lambda x:np.isclose(x[2], 0.0))   
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    f = fem.Function(V)
    f.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    energy=fem.form(u* ufl.dx)
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()    
    value=fem.assemble.assemble_scalar(energy)
    if write:
        with io.XDMFFile(domain.comm, name+".xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
    return value

if __name__=="__main__":
    mymesh=meshio.read("Stanford_Bunny_red.stl")
    print(np.min(mymesh.points))
    bary=np.mean(mymesh.points,axis=0)
    calculate_simulation("Stanford_Bunny_red",bary)

