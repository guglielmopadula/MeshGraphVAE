import meshio
import tetgen
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import numpy as np
from tqdm import trange
NUM_SAMPLES=600
mymesh=meshio.read("data/bunny_{}.stl".format(0))
tgen = tetgen.TetGen(mymesh.points,mymesh.cells_dict["triangle"])
nodes, elem = tgen.tetrahedralize(quality=False,nobisect=True)

print(np.min(mymesh.points))
bary=np.mean(mymesh.points,axis=0)
pts_ref=mymesh.points.copy().reshape(-1)

index=np.random.choice(len(pts_ref), 50, replace=False)
pts_ref_train=pts_ref[index].reshape(-1,1)
pts_ref_test=np.delete(pts_ref,index).reshape(-1,1)
pts_ref=pts_ref.reshape(-1,1)
kernel = DotProduct() + WhiteKernel()

for i in trange(1,NUM_SAMPLES):
    mymesh=meshio.read("data/bunny_{}.stl".format(i))
    pts=mymesh.points.reshape(-1)
    pts_train=pts[index].reshape(-1,1)
    pts_test=np.delete(pts,index).reshape(-1,1)
    pts=pts.reshape(-1,1)
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(pts_ref_train,pts_ref_train )
    print(np.linalg.norm(gpr.predict(pts_ref_test)-pts_test))
