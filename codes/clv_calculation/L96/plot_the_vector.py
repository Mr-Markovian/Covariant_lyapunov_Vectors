"""We plot the mean of cosines of the angles after the BLVs converge.
We also then plot the mean versus clv index for different values of sigma.
We also obtain the exponents and their values"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

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
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/seed_{}'.format(model_dim,seed_num)
os.chdir(data_path)

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('sigma=0.0')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]
os.chdir('..')
# save these 

for i in range(20):
    plt.matshow(np.log(np.abs(V[i])))
    plt.title('CLV_t={}'.format(i*0.05))
    plt.savefig('CLV_t={}.png'.format(i*0.05))
#plt.show()


# We have first 10000 time steps as the transient to converge to BLVs.

# sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
# for sigma in sigmas:
#     os.chdir(data_path+'/sigma={}'.format(sigma))
#     C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
#     G1=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
#     V1=np.zeros_like(G1)
#     for i in range(G1.shape[0]):
#         V1[i]=G1[i]@C1[i]

#     # Angle between BLVs
#     cosines=np.zeros((G.shape[0],num_clv))
#     for i in range(G.shape[0]):
#         for j in range(num_clv):
#             cosines[i,j]=np.absolute(np.dot(G1[i,:,j],G[i,:,j]))

#     # Angle between CLVs
#     cosines2=np.zeros((G.shape[0],num_clv))
#     for i in range(G.shape[0]):
#         for j in range(num_clv):
#             cosines2[i,j]=np.absolute(np.dot(V1[i,:,j],V[i,:,j]))

    

