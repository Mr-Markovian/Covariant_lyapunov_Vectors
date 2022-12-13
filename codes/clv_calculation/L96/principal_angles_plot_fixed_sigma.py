import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=20
mpl.rcParams['legend.fontsize']=15
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15

dt=0.05   
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-40/seed_3'
os.chdir(data_path+'/P-angles_sigma=0.3')

plt.figure(figsize=(16,8))
sigmas=np.array([0.1,0.3,0.5])
line_types=['dotted','dashed','solid']

subspace_nums=20
chosen_subspaces=np.array([2,5,10,15,20])

for i,sigma in enumerate(sigmas):
    for subspace_num in chosen_subspaces:
        clv_cosines=np.load('blv_principal_cosine_sigma={}_model_dim={}_subspaces_={}.npy'.format(sigma,model_dim,subspace_num))
        quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
        asymmetric_bar=np.zeros((2,subspace_num))
        asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
        asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
        #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
        plt.errorbar(x=np.arange(1,subspace_num+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,ls=line_types[i],label=r'$n={}$'.format(subspace_num),marker='.',ms=20)



plt.title(r'P.A. for different n-dimensional subspace, $\sigma=0.3$')
plt.xlabel(r'index $\to$')
plt.ylim(0.0,1.1)
plt.yticks(np.arange(0,100,10))
plt.xticks(np.arange(1,subspace_num+1))
#plt.yticks()
plt.ylabel(r'$ \theta \to$')
#plt.legend(loc='upper center',ncol=subspace_nums )
plt.savefig('blv_principals_3_point_summary_angle_sensitivity_L96-{}_analysis_subspaces_{}.png'.format(model_dim,subspace_num))



