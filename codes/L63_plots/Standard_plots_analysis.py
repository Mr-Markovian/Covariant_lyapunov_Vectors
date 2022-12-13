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
base_type1='Analysis'
mu=1.0
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/Analysis')
C2=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type1))[t_start:t_stop]
G2=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type1))[t_start:t_stop]
analysis=np.load('Analysis_mean_g={}_mu={}.npy'.format(dt,mu))[start_idx+t_start:start_idx+t_stop]


V2=np.zeros_like(G)
for i in range(G.shape[0]):
    V2[i]=G2[i]@C2[i]    

plt.figure(figsize=(16,8))
comp_=1
plt.plot(0.01*np.arange(base_traj.shape[0])+350,np.linalg.norm(base_traj-analysis,axis=1),label='analysis_error')
plt.xlabel(r'$t \to$')
plt.title('L2 error in analysis.png')
plt.savefig('analysis_l2error.png'.format(comp_,mu))


# Plot the clv for noisy and true trajectory 
num_clv=2
spr=2
plot_pairs=[[0,1],[1,2]]
for clv_index in range(num_clv):
    for l,m in plot_pairs:
        fig, ax = plt.subplots(figsize=(16,8))
        ax.quiver(base_traj[::spr,l],base_traj[::spr,m],V[::spr,l,clv_index],V[::spr,m,clv_index],scale_units='xy',scale=1.0,color='black',label='{}'.format(base_type))
        ax.quiver(base_traj[::spr,l],base_traj[::spr,m],V2[::spr,l,clv_index],V2[::spr,m,clv_index],scale_units='xy',scale=1.0,color='blue',label='{}'.format(base_type1),alpha=0.6)        
        ax.scatter(base_traj[0,l],base_traj[0,m],c='orange',s=100,edgecolors='blue')
        #ax.scatter(traj[0,l],traj1[0,m],c='r',s=80,edgecolors='black')
        ax.set_title('CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
        ax.set_xlabel('{}-axis'.format(coord[l]))
        ax.set_ylabel('{}-axis'.format(coord[m]))
        plt.legend()
        plt.tight_layout()
        plt.savefig('CLV{} in {}{}for_{}.png'.format(clv_index+1,coord[l],coord[m],base_type1))

# To check, average errors over the interval for which clv is calculated. And the cosine of the angle between them
# and the 
l2_error=np.arange(base_traj.shape[0])
l2_error=np.linalg.norm(analysis-base_traj,axis=1)
cosines=np.zeros((base_traj.shape[0],3))
for i in range(base_traj.shape[0]):
    for j in range(3):
        cosines[i,j]=np.absolute(np.dot(V[i,:,j],V2[i,:,j]))

plt.figure(figsize=(16,4))
colors=['r','g','b']
plt.subplot(1,2,1)
for i in range(3):
    sns.histplot(x=cosines[:,i],fill=True,color=colors[i],label='{}'.format(i))
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
for i in range(3):
    plt.plot(np.arange(len(l2_error)),cosines[:,i],label='{}'.format(i),lw=0.8,c=colors[i])
plt.xlabel(r'$t \to$')
plt.legend()
plt.tight_layout()
plt.savefig('CLV_angles_time_and_dist.png')

#plot the cosine of the angle over the trajectory
fig = plt.figure(figsize=(14,14))
comp_=0
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
# plotting
my_scatter=ax.scatter(base_traj[:,0],base_traj[:,1],base_traj[:,2],c=cosines[:,comp_],label='{}'.format(base_type),cmap='viridis')
ax.set_xlabel(r'$X\to$')
ax.set_ylabel(r'$Y\to$')
ax.set_zlabel(r'$Z\to$')
fig.colorbar(my_scatter,shrink=0.5, aspect=2)
plt.title('Cosine over attractor')
plt.legend()
plt.savefig('cosine_error_over_att_comp={}.png'.format(comp_))