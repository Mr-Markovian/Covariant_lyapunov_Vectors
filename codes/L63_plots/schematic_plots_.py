# Schematic plots for thesis:

# Plot one: showing how two different ensembles converge 

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns
from cmcrameri import cm
plt.style.use('tableau-colorblind10')

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

# 1st assim experiment
# Change here for different observation gap and observation covariance parameter
mu=0.5
ob_gap=0.05
k=4
N=40 
loc_fun='gaspri'
l_scale=4
alpha=1.0
ob_dim=20
ecov=1.0 

ebias=4.0     
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96')

#load the state
state=np.load('state/Multiple_trajectories_N=1_gap=0.05_ti=0.0_tf=500.0_dt_0.05_dt_solver=0.01.npy')
# State and obs
os.chdir(os.getcwd()+'/assimilation/ob{}'.format(k)) 
obs=np.load('ob{}_gap_{}_H2__mu={}_obs_cov1.npy'.format(k,ob_gap,mu))
#Go inside the data folder......................................
folder_label='ebias={}_ecov={}_obs={}_ens={}_mu={}_gap={}_alpha={}_loc=gaspri_r={}'.format(ebias, ecov,ob_dim,N,mu,ob_gap,alpha,l_scale)
print(os.getcwd())
os.chdir(folder_label)
#Load data....
a_ens=np.load('filtered_ens.npy') #ens has shape:=[time steps,system dimension,ensemble number]
a_mean=np.mean(a_ens,axis=2)


# load the second assim exp
# Change here for different observation gap and observation covariance parameter
mu=0.5
ob_gap=0.05
k=4
N=40 
loc_fun='gaspri'
l_scale=4
alpha=1.0
ob_dim=20
ecov=2.0 
ebias=-5.0     



os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96')

#load the state
state=np.load('state/Multiple_trajectories_N=1_gap=0.05_ti=0.0_tf=500.0_dt_0.05_dt_solver=0.01.npy')


# State and obs
os.chdir(os.getcwd()+'/assimilation/ob{}'.format(k)) 
obs=np.load('ob{}_gap_{}_H2__mu={}_obs_cov1.npy'.format(k,ob_gap,mu))

#Go inside the data folder......................................
folder_label='ebias={}_ecov={}_obs={}_ens={}_mu={}_gap={}_alpha={}_loc=gaspri_r={}'.format(ebias, ecov,ob_dim,N,mu,ob_gap,alpha,l_scale)
print(os.getcwd())
os.chdir(folder_label)

#Load data....
a1_ens=np.load('filtered_ens.npy') #ens has shape:=[time steps,system dimension,ensemble number]
a1_mean=np.mean(a1_ens,axis=2)

# Now plots
time=ob_gap*np.arange(a_mean.shape[0])
plt.figure(figsize=(16,8))
# Start time and end time chosen to view a part of time series
t_start=0
t_stop=400
# component to view
comp_=5
#plt.plot(time[t_start:t_stop],a1_mean[t_start:t_stop,comp_],c='r',alpha=1)

plt.plot(time[t_start:t_stop],a_ens[t_start:t_stop,comp_],c='r',alpha=0.2)
plt.plot(time[t_start:t_stop],a1_ens[t_start:t_stop,comp_],c='blue',alpha=0.2)
plt.plot(time[t_start:t_stop],state[t_start:t_stop,comp_],c='black',label='State')
if (comp_%2==0):
    #plt.scatter(time[t_start:t_stop],obs[t_start:t_stop,int(comp_/2)],c='black',edgecolors='black',marker='.',s=150,label='obs')
    plt.errorbar(x=time[t_start:t_stop],y=obs[t_start:t_stop,int(comp_/2)],yerr=mu,c='g',mec='black',mfc='g',capsize=4,fmt='.',ms=13,label='observations')

plt.legend()
#plt.ylabel(r'$X_{}$'.format(comp_+1))
plt.xlim(0,10)
plt.xticks([])
plt.yticks([])
plt.xlabel(r'time(n) $\to$')
plt.text(-0.4,10,r'$\nu$',fontsize=20)
plt.text(-0.4,0,r'$\mu$',fontsize=20)
plt.text(-0.4,6,r'$x_o$',fontsize=20)
plt.text(8.5,8,r'$\pi_n(\nu) \approx \pi_n(\mu)$',fontsize=20)


#plt.xticks(time[t_start:t_stop],fontsize=12)
plt.legend(frameon='True')
os.chdir('/home/shashank/Dropbox/Data_Assimilation/Thesis_writing/figures')
plt.savefig('stabilit_enkf_picture_x{}.png'.format(comp_))


