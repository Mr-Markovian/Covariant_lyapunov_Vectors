import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

dt=0.01   
model_dim=3
model_name='L63'
coord=['X','Y','Z']

# making the state noisy
spr=1
t_start=0
t_stop=5000

# Change to data path
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
start_idx=35000 # starting point of the interval of clv( 25000+10000)
base_traj=np.load('State_g={}.npy'.format(dt))[start_idx:start_idx+t_stop]

# CLVs about the true state
base_type='State'
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]

print(C.shape)
print(G.shape)
print(base_traj.shape)
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  


# CLVs about the true state
mu=1.0
base_type='state_noisy'
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/mu={}'.format(mu))
C2=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G2=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
noisy_traj=np.load('{}_g={}_mu={}.npy'.format(base_type,dt,mu))[10000+t_start:10000+t_stop]

print(C2.shape)
print(G2.shape)
print(noisy_traj.shape)
V2=np.zeros_like(G)
for i in range(G.shape[0]):
    V2[i]=G2[i]@C2[i]    


# plt.figure(figsize=(16,8))
# comp_=1
# plt.plot(np.arange(base_traj.shape[0]),noisy_traj[:,comp_],lw=0.5,label='Noisy')
# plt.plot(np.arange(base_traj.shape[0]),base_traj[:,comp_],c='black',label='Truth')
# plt.xlabel(r'$t \to$')
# plt.title('Noisy trajectory for mu={}'.format(mu))
# plt.savefig('noisy_state_comp={}_mu={}.png'.format(comp_,mu))

cosines=np.zeros((base_traj.shape[0],3))
for i in range(base_traj.shape[0]):
    for j in range(3):
        cosines[i,j]=np.absolute(np.dot(V[i,:,j],V2[i,:,j]))

plt.figure(figsize=(16,4))
colors=['r','g','b']
for i in range(3):
    sns.histplot(x=np.log(cosines[:,i]),fill=True,color=colors[i],label='{}'.format(i))
plt.legend()
plt.tight_layout()

plt.savefig('CLV_angles_time_and_dist_mu={}.png'.format(mu))
