from nn.models.losses.losses import mmd
import numpy as np
import matplotlib.pyplot as plt
names=["AE","VAE","AAE","BEGAN","EBM","DM","NF"]
db_t=["u","energy"]
approximations =  [
    'RBF',
    'GPR',
    'KNeighbors',
    'ANN'
]

var_tot=np.zeros(8)
mmd_tensor_tot=np.zeros(7)
mmd_area_tot=np.zeros(7)
mmd_id_tot=np.zeros(7)
rec_error_tot=np.zeros(7)
mmd_energy_tot=np.zeros(7)
mmd_u_tot=np.zeros(7)

train_energy_tot=np.zeros((8,4))
test_energy_tot=np.zeros((8,4))
train_u_tot=np.zeros((8,4))
test_u_tot=np.zeros((8,4))



for i in range(len(names)):
    name=names[i]
    moment_tensor_data=np.load("nn/geometrical_measures/moment_tensor_data.npy")
    moment_tensor_sampled=np.load("nn/geometrical_measures/moment_tensor_"+name+".npy")
    area_data=np.load("nn/geometrical_measures/area_data.npy")
    area_sampled=np.load("nn/geometrical_measures/area_"+name+".npy")
    variance=np.load("nn/geometrical_measures/variance_"+name+".npy")
    variance_data=np.load("nn/geometrical_measures/variance_data.npy")
    mmd_data=np.load("nn/geometrical_measures/mmd_"+name+".npy")
    error=np.load("nn/geometrical_measures/rel_error_"+name+".npy")
    energy_data=np.load("simulations/data/energy_data.npy")
    energy_sampled=np.load("simulations/inference_objects/energy_"+name+".npy")
    mmd_u=np.load("simulations/inference_objects/mmd_u_"+name+".npy")
    perc_pass=np.load("nn/geometrical_measures/perc_pass_"+name+".npy")
    train_error_rom_sampled=np.load("./simulations/inference_objects/"+name+"_rom_err_train.npy")
    test_error_rom_sampled=np.load("./simulations/inference_objects/"+name+"_rom_err_test.npy")
    train_error_rom_data=np.load("./simulations/inference_objects/data_rom_err_train.npy")
    test_error_rom_data=np.load("./simulations/inference_objects/data_rom_err_test.npy")
    var_tot[0]=variance_data.item()
    mmd_area_tot[i]=mmd(area_data,area_sampled)
    mmd_tensor_tot[i]=mmd(moment_tensor_data.reshape(-1,np.prod(moment_tensor_data.shape[1:])),moment_tensor_sampled.reshape(-1,np.prod(moment_tensor_data.shape[1:])))
    var_tot[i+1]=variance.item()
    rec_error_tot[i]=error.item()
    mmd_id_tot[i]=mmd_data
    mmd_energy_tot[i]=mmd(energy_data,energy_sampled)
    mmd_u_tot[i]=mmd_u

    for j in range(len(approximations)):
        train_u_tot[0,j]=train_error_rom_data[0,j]
        test_u_tot[0,j]=test_error_rom_data[0,j]
        train_energy_tot[0,j]=train_error_rom_data[1,j]
        test_energy_tot[0,j]=test_error_rom_data[1,j]

        train_u_tot[i+1,j]=train_error_rom_sampled[0,j]
        train_energy_tot[i+1,j]=train_error_rom_sampled[1,j]
        test_u_tot[i+1,j]=test_error_rom_sampled[0,j]
        test_energy_tot[i+1,j]=test_error_rom_sampled[1,j]



    fig2,ax2=plt.subplots()
    ax2.set_title("Area of "+name)
    _=ax2.hist([area_data,area_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/Area_hist_"+name+".pdf")
    fig2,ax2=plt.subplots()
    ax2.set_title("Energy of "+name)
    _=ax2.hist([energy_data,energy_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/Energy_hist_"+name+".pdf")
#Geometrical quantities
fig2,ax2=plt.subplots()
ax2.set_title("MMD between moment tensor of data and of GM")
ax2.plot(names,mmd_tensor_tot)
fig2.savefig("./inference_graphs_txt/Moment.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("MMD between area of data and of GM")
ax2.plot(names,mmd_area_tot)
fig2.savefig("./inference_graphs_txt/Area.pdf")
#Physical quantities
fig2,ax2=plt.subplots()
ax2.set_title("MMD between energy of data and of GM")
ax2.plot(names,mmd_energy_tot)
fig2.savefig("./inference_graphs_txt/Energy.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("MMD between u of data and of GM")
ax2.plot(names,mmd_u_tot)
fig2.savefig("./inference_graphs_txt/u.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("MMD between Id of data and of GM")
ax2.plot(names,mmd_id_tot)
fig2.savefig("./inference_graphs_txt/Id.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("Rec error between data and GM")
ax2.plot(names,rec_error_tot)
fig2.savefig("./inference_graphs_txt/rec.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("Variance")
ax2.plot(["data"]+names,var_tot)
fig2.savefig("./inference_graphs_txt/var.pdf")

fig2,ax2=plt.subplots()
ax2.set_title("ROM u train error")

for j in range(len(approximations)):
    ax2.plot(["data"]+names,train_u_tot[:,j],label=approximations[j])
    
ax2.legend()
fig2.savefig("./inference_graphs_txt/train_u.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM u test error")

for j in range(len(approximations)):
    ax2.plot(["data"]+names,test_u_tot[:,j],label=approximations[j])
    
ax2.legend()
fig2.savefig("./inference_graphs_txt/test_u.pdf")

fig2,ax2=plt.subplots()
ax2.set_title("ROM energy train error")

for j in range(len(approximations)):
    ax2.plot(["data"]+names,train_energy_tot[:,j],label=approximations[j])
    
ax2.legend()
fig2.savefig("./inference_graphs_txt/train_energy.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM energy test error")

for j in range(len(approximations)):
    ax2.plot(["data"]+names,test_energy_tot[:,j],label=approximations[j])
    
ax2.legend()
fig2.savefig("./inference_graphs_txt/test_energy.pdf")
