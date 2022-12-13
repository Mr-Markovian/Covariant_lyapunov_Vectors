import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlow
colors = [colormap(0), colormap(100),colormap(200)]

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=16
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

dt=0.01   
dt_solver=0.005
model_dim=3
model_name='L63'
num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63/seed_{}'.format(seed_num)

os.chdir(data_path)
#sigmas=np.array([0.01,0.05,0.1,0.2,0.4,1.0,2.0,4.0])
sigmas=np.array([1.0,2.0,3.0,4.0,5.0])

bars=np.zeros((2,len(sigmas),num_clv))
medians=np.zeros((len(sigmas),num_clv))

for i,sigma in enumerate(sigmas):
    clv_cosines=np.rad2deg(np.arccos(np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
    quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
    bars[0,i,:]=quantiles_[1]-quantiles_[0]
    bars[1,i,:]=quantiles_[2]-quantiles_[1]
    medians[i]=np.median(clv_cosines,axis=0)


#plt.figure(figsize=(16,8))   

fig, ax =plt.subplots(figsize=(16,8))  
for i in range(num_clv):     
    ax.errorbar(x=sigmas,y=medians[:,i],yerr=bars[:,:,i],capsize=6,mec='black',label=r'$CLV$={}'.format(i+1),marker='.',ms=20,c=colors[i])

ax.set_xlabel(r'$\sigma \to$')
ax.set_ylim(0.0,80)
ax.set_xticks(sigmas)
ax.set_ylabel(r'$\theta \to$')
#plt.legend(loc='lower center',ncol=len(sigmas))
plt.legend(loc='upper right')

# calculate data for the inset
sigmas1=np.array([0.1,0.2,0.3,0.4,0.5])
bars1=np.zeros((2,len(sigmas1),num_clv))
medians1=np.zeros((len(sigmas1),num_clv))

for i,sigma in enumerate(sigmas1):
    clv_cosines=np.rad2deg(np.arccos(np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
    quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
    bars1[0,i,:]=quantiles_[1]-quantiles_[0]
    bars1[1,i,:]=quantiles_[2]-quantiles_[1]
    medians1[i]=np.median(clv_cosines,axis=0)

ax_inset = ax.inset_axes([0.05, 0.5, 0.6, 0.5])#)
ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
for i in range(num_clv):     
    ax_inset.errorbar(x=sigmas1,y=medians1[:,i],yerr=bars1[:,:,i],capsize=4,mec='black',label=r'$CLV$={}'.format(i+1),marker='.',ms=20,c=colors[i])

ax_inset.set_xticks(sigmas1)
#ax_inset.set_ylabel('')


plt.savefig('clv_angle_versus_sigma_L63-{}_seed_{}.png'.format(model_dim,seed_num))


# for sigma in sigmas:
#     clv_cosines=np.rad2deg(np.arccos(np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
#     # plt.plot(np.arange(1,num_clv+1),np.mean(clv_cosines,axis=0))
#     # plt.errorbar(x=np.arange(1,num_clv+1),y=np.mean(clv_cosines,axis=0),yerr=np.std(clv_cosines,axis=0),capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)

#     #clv_cosines=np.rad2deg(np.arccos(np.load('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
#     quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
#     asymmetric_bar=np.zeros((2,num_clv))
#     asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
#     asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
#     #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
#     plt.errorbar(x=np.arange(1,num_clv+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)


# plt.xlabel(r'LV-index $\to$')
# #plt.yticks(np.arange(0.3,1.1,0.1))
# plt.xticks(np.arange(1,num_clv+1))
# plt.ylabel(r'$\theta \to$')
# #plt.legend(loc='lower center',ncol=len(sigmas))
# plt.legend(loc='center right')
# plt.savefig('clv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))
# #plt.savefig('blv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))

# plt.figure(figsize=(16,8))
# for sigma in sigmas:
#     lexp=np.load('exp_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
#     plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(sigma),s=20)

# plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(0.0),s=20,c='black')
# plt.xlabel(r'index $\to$')
# plt.ylabel(r'$\hat \lambda $')
# plt.xticks(np.arange(1,num_clv+1))
# plt.legend()
# plt.savefig('exponent_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))


