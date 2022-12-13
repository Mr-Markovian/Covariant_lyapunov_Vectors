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

  
os.chdir(plots_path)

plot_pairs=[[0,1],[1,2],[0,2]]
for clv_index in range(num_clv):
    for l,m in plot_pairs:
        fig, ax = plt.subplots(figsize=(16,8))
        ax.quiver(base_traj[::spr,l],base_traj[::spr,m],V[::spr,l,clv_index],V[::spr,m,clv_index],scale_units='xy',scale=1.0,color='black',label='{}'.format(base_type))
        ax.scatter(base_traj[0,l],base_traj[0,m],c='r',s=80,edgecolors='black')
        ax.set_title('CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
        ax.set_xlabel('{}-axis'.format(coord[l]))
        ax.set_ylabel('{}-axis'.format(coord[m]))
        plt.legend()
        plt.tight_layout()
        plt.savefig('CLV{} in {}{}for_{}.png'.format(clv_index+1,coord[l],coord[m],'truth'))

print('Job done')

# # Plotting Forward vectors
# for clv_index in range(num_clv):
#     for l,m in plot_pairs:
#         fig, ax = plt.subplots(figsize=(16,8))
#         ax.quiver(traj[::spr,l],traj[::spr,m],G[::spr,clv_index,l],G[::spr,clv_index,m],scale_units='xy',scale=1.0,color='black',label='trajectory 1')
#         ax.scatter(traj[0,l],traj[0,m],c='r',s=50,edgecolors='black')
#         ax.quiver(traj2[::spr,l],traj2[::spr,m],G2[::spr,clv_index,l],G2[::spr,clv_index,m],scale_units='xy',scale=1.0,color='blue',label='trajectory 2')
#         ax.scatter(traj2[0,l],traj2[0,m],c='orange',s=70,edgecolors='blue')
#         ax.set_title('FLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
#         ax.set_xlabel('{}-axis'.format(coord[l]))
#         ax.set_ylabel('{}-axis'.format(coord[m]))
#         plt.legend()
#         plt.savefig('FLV{} in {}{}place_seed_{}.png'.format(clv_index+1,coord[l],coord[m],seed_num))


#fig = plt.figure(figsize=(16,8))
#ax = plt.axes(projection ='3d')
# ax.plot3D(traj[:,0], traj[:,1], traj[:,2],label='trajectory-1',c='black')
# ax.plot3D(traj2[:,0],traj2[:,1],traj2[:,2],label='trajectory-2')
# ax.set_title('Attractor',fontsize=18)
# ax.set_xlabel(r'$X\to$')
# ax.set_ylabel(r'$Y\to$')
# ax.set_zlabel(r'$Z\to$')
# plt.legend()
# plt.savefig('trajectory.png')
#Quiver plots for the different clvs..

# just check if the vectors are of magnitude 1
norms=np.sum(np.sum(V**2,axis=-1),axis=1)
