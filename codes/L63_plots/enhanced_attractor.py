import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlowK

# A colormap in dark mode
#colormap=cm.vanimo

data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63/seed_3'
plots_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/images/L63'


mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

spr=1
dt=0.01   
t_start=0
t_stop=5000

model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
coord=['X','Y','Z']

os.chdir(data_path)
os.chdir('sigma=0.0')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]

#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
start_idx=10000 # starting point of the interval of clv( 25000+10000)
base_traj=np.load('{}_g={}_sigma=0.0.npy'.format(base_type,dt))[start_idx:start_idx+t_stop]

V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  

# Cosines between the 1st and 2nd clv
cosines=np.zeros((base_traj.shape[0]))
for i in range(base_traj.shape[0]):
        cosines[i]=np.dot(V[i,:,0],V[i,:,1])

# Regime change in L63, by using CLVs
#plot the state and the trajectory

os.chdir(plots_path)
fig = plt.figure(figsize=(16,12))
  
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
# plotting
my_scatter=ax.scatter(base_traj[:,0],base_traj[:,1],base_traj[:,2],c=cosines,label='{}'.format(base_type),cmap=colormap)
ax.set_xlabel(r'$X\to$')
ax.set_ylabel(r'$Y\to$')
ax.set_zlabel(r'$Z\to$')
ax.set_title('cosine between clv{} and clv{}'.format(1,2))
fig.colorbar(my_scatter,shrink=0.5, aspect=5)
plt.legend()
plt.savefig('enhanced_L63_attr.png',dpi=200, bbox_inches='tight', pad_inches=0)