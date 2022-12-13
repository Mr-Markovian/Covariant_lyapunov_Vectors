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
model_dim=10
model_name='L96'
num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/seed_{}'.format(model_dim,seed_num)
os.chdir(data_path)

#Load the data for the actual trajectory

# We have first 10000 time steps as the transient to converge to BLVs

#Some customization for the box plots
meanpointprops=dict(marker='D',linestyle='-',linewidth=4,c='r')
medianpointprops=dict(marker='.',linestyle='-',linewidth=4,color='black')

sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
for sigma in sigmas:
    cosines=np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))

    # save these 
    os.chdir(data_path)
    plt.figure(figsize=(16,8))
    plt.boxplot(x=cosines,positions=np.arange(1,num_clv+1),notch=True,medianprops=medianpointprops,showmeans=True,meanprops=meanpointprops,patch_artist=True,showfliers=False)
    plt.ylim(-0.1,1.1)
    plt.ylabel(r'cosine $(\theta)$')
    plt.xlabel(r'index $\to$')
    plt.title('CLV for sigma={}'.format(sigma))
    plt.yticks(np.arange(0.0,1.1,0.1))
    plt.savefig('clv_box_sigma_{}.png'.format(sigma))

