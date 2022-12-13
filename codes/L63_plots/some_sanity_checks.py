# Some sanity checks
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63/seed_3'
plots_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/images/L63'

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

spr=2
model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
coord=['X','Y','Z']
dt=0.01
mu=0.0
t_start=0
t_stop=5000
start_idx=10000

# vectors for the true trajectory
os.chdir(data_path+'/sigma={}'.format(0.0))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
base_traj=np.load('{}_g={}_sigma={}.npy'.format(base_type,dt,0.0))[start_idx+t_start:start_idx+t_stop]

#print(C.shape,G.shape,traj.shape)
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]


# just check if the vectors are of magnitude 1
norms=np.sum(np.sum(V**2,axis=-1),axis=1)
indices=np.random.randint(0,3000,50)
for i in range(50):
    print(norms[indices[i]])