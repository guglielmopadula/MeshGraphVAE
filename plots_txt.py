from nn.models.losses.losses import mmd
import numpy as np
import matplotlib.pyplot as plt
names=["AE","VAE","AAE","BEGAN","EBM","DM","NF"]
db_t=["u","energy"]
approximations =  [
    'RBF',
    'GPR',
    'KNeighbors',
    'RadiusNeighbors',
    'ANN'
]

for name in names:


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


    f = open("./inference_graphs_txt/"+name+"_sampled.txt", "a")
    f.write("MMD Area distance of"+name+" is "+str(mmd(area_data,area_sampled))+"\n")
    print("MMD Area distance of"+name+" is "+str(mmd(area_data,area_sampled))+"\n")
    f.write("MMD Moment distance of"+name+" is "+str(mmd(moment_tensor_data.reshape(-1,np.prod(moment_tensor_data.shape[1:])),moment_tensor_sampled.reshape(-1,np.prod(moment_tensor_data.shape[1:]))))+"\n")
    print("MMD Moment distance of"+name+" is "+str(mmd(moment_tensor_data.reshape(-1,np.prod(moment_tensor_data.shape[1:])),moment_tensor_sampled.reshape(-1,np.prod(moment_tensor_data.shape[1:]))))+"\n")
    f.write("Percentage error "+name+" is "+str(error.item())+"\n")
    print("Percentage error "+name+" is "+str(error.item())+"\n")
    f.write("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    print("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    f.write("MMD Id distance of"+name+" is "+str(mmd_data)+"\n")
    print("MMD Id distance of"+name+" is "+str(mmd_data)+"\n")
    f.write("MMD Energy distance of"+name+" is "+str(mmd(energy_data,energy_sampled))+"\n")
    print("MMD Energy distance of"+name+" is "+str(mmd(energy_data,energy_sampled))+"\n")
    f.write("MMD u distance of"+name+" is "+str(mmd_u)+"\n")
    print("MMD u distance of"+name+" is "+str(mmd_u)+"\n")
    print("Percentage of passing samples of " + name + " is " + str(perc_pass)+"\n")
    f.write("Percentage of passing samples of " + name + " is " + str(perc_pass)+"\n")
    for i in range(len(db_t)):
        for j in range(len(approximations)):
            print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error_rom_sampled[i,j])+"\n")
            f.write("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error_rom_sampled[i,j])+"\n")
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error_rom_sampled[i,j])+"\n")
            f.write("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error_rom_sampled[i,j])+"\n")




    f.close()
    fig2,ax2=plt.subplots()
    ax2.set_title("XX moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,0].reshape(-1),moment_tensor_sampled[:,0,0].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/XXaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,1].reshape(-1),moment_tensor_sampled[:,1,1].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/YYaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("ZZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,2,2].reshape(-1),moment_tensor_sampled[:,2,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/ZZaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,1].reshape(-1),moment_tensor_sampled[:,0,1].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/XYaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,2].reshape(-1),moment_tensor_sampled[:,0,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/XZaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,2].reshape(-1),moment_tensor_sampled[:,1,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/YZaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("Area of "+name)
    _=ax2.hist([area_data,area_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/Area_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("Energy of "+name)
    _=ax2.hist([energy_data,energy_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs_txt/Energy_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    

f = open("./inference_graphs_txt/data.txt", "a")
for i in range(len(db_t)):
    for j in range(len(approximations)):
        print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error_rom_data[i,j])+"\n")
        f.write("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error_rom_data[i,j])+"\n")
        print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error_rom_data[i,j])+"\n")
        f.write("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error_rom_data[i,j])+"\n")

