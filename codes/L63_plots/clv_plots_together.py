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

spr=3

model_dim=3
model_name='L63'
base_type2='state_noisy'
base_type1='state_noisy'
num_clv=3
coord=['X','Y','Z']
dt=0.01
sigma=1.0
t_start=0
t_stop=5000
start_idx=10000


# vectors for the true trajectory
os.chdir(data_path+'/sigma={}'.format(0.0))
C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type1))[t_start:t_stop]
G1=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type1))[t_start:t_stop]
base_traj1=np.load('{}_g={}_sigma={}.npy'.format(base_type1,dt,0.0))[start_idx+t_start:start_idx+t_stop]

# vectors for the noisy trajectory
os.chdir(data_path+'/sigma={}'.format(sigma))
C2=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type2))[t_start:t_stop]
G2=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type2))[t_start:t_stop]
base_traj2=np.load('{}_g={}_sigma={}.npy'.format(base_type1,dt,sigma))[start_idx+t_start:start_idx+t_stop]

#print(C.shape,G.shape,traj.shape)
V1=np.zeros_like(G1)
for i in range(G1.shape[0]):
    V1[i]=G1[i]@C1[i]

V2=np.zeros_like(G2)
for i in range(G2.shape[0]):
     V2[i]=G2[i]@C2[i]    


os.chdir(plots_path)

# plot the vectors in xy, yz and xz planes.
plot_pairs=[[0,1],[1,2],[0,2]]
for clv_index in range(num_clv):
    for l,m in plot_pairs:
        fig, ax = plt.subplots(figsize=(16,8))
        ax.quiver(base_traj1[::spr,l],base_traj1[::spr,m],V1[::spr,l,clv_index],V1[::spr,m,clv_index],angles='xy',scale_units='xy',scale=1.0,color='black',headwidth=1.5,label='truth')
        ax.quiver(base_traj1[::spr,l],base_traj1[::spr,m],V2[::spr,l,clv_index],V2[::spr,m,clv_index],angles='xy',scale_units='xy',scale=1.0,color='royalblue',headwidth=1.5,label=r'$\sigma={}$'.format(sigma))
        ax.scatter(base_traj1[0,l],base_traj1[0,m],c='r',s=80,edgecolors='black')
        #ax.scatter(base_traj1[0,l],base_traj1[0,m],c='g',s=80,edgecolors='black')        
        ax.set_title('CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
        ax.set_xlabel('{}-axis'.format(coord[l]))
        ax.set_ylabel('{}-axis'.format(coord[m]))
        plt.legend()
        plt.tight_layout()
        plt.savefig('CLV{} in {}{}for_both.png'.format(clv_index+1,coord[l],coord[m]))

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

print('Job done')
