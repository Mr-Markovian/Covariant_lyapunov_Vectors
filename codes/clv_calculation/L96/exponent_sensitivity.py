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
model_dim=20
model_name='L96'
num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/seed_{}'.format(model_dim,seed_num)
os.chdir(data_path)

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('sigma=0.0')
exp=np.load('local_growth_rates_{}_model_L96_State.npy'.format(model_dim))
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.

plt.figure(figsize=(16,8))
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
for i,sigma in enumerate(sigmas):
    os.chdir(data_path+'/sigma={}'.format(sigma))
    exp1=np.load('local_growth_rates_{}_model_L96_state_noisy.npy'.format(model_dim))
    plt.scatter(np.arange(1,num_clv+1),np.mean(exp1,axis=0),label='$\sigma={}$'.format(sigma),marker='^',s=30)
    # save these 
    #os.chdir(data_path)
    #store_lexp=np.mean(exp1,axis=0)
    #np.save('exp_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv),store_lexp)
plt.scatter(np.arange(1,num_clv+1),np.mean(exp,axis=0),label='$truth$',marker='o',s=20)
plt.xlabel(r'index $ \to$')
plt.ylabel(r'$\lambda_i$')
plt.legend(ncol=len(sigmas))
os.chdir(data_path)
plt.savefig('L96_{}_exp_sensitivity_seed_{}.png'.format(model_dim,seed_num))