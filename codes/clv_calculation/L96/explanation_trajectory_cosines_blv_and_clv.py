"""We calculate the cosines of the angles for the assimialted trajectories with the observation
noise of the form mu times Identity."""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from scipy.linalg import svdvals

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

dt=0.05  
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/explanation'.format(model_dim)
os.chdir(data_path)

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('analysis_sigma=0.0')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
for sigma in sigmas:
    folder_name='analysis_sigma={}'.format(sigma)
    os.chdir(data_path+'/'+folder_name)
    C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    G1=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    V1=np.zeros_like(G1)
    for i in range(G1.shape[0]):
        V1[i]=G1[i]@C1[i]

    print(G1.shape)
    # Angle between BLVs
    cosines=np.zeros((G.shape[0],num_clv))
    for i in range(G.shape[0]):
        for j in range(num_clv):
            cosines[i,j]=np.absolute(np.dot(G1[i,:,j],G[i,:,j]))

    # Angle between CLVs
    cosines2=np.zeros((G.shape[0],num_clv))
    for i in range(G.shape[0]):
        for j in range(num_clv):
            cosines2[i,j]=np.absolute(np.dot(V1[i,:,j],V[i,:,j]))

    # Principals of the two subspaces
    subspace_num=2
    cosines3=np.zeros((G.shape[0],subspace_num))
    for i in range(G.shape[0]):
        D=svdvals(G1[i,:,:subspace_num].T@G[i,:,:subspace_num])
        cosines3[i]=D[:subspace_num]

    # save these 
    os.chdir(data_path)

    np.save('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv),cosines)
    np.save('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv),cosines2)
    np.save('blv_principal_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv),cosines3)
    #store_lexp=np.mean(exp1,axis=0)
    #np.save('exp_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv),store_lexp)

