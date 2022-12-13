import os
from attr import s
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib as mpl
from numpy.linalg.linalg import eigvals
import seaborn as sns

# Codes for generating final 
plt.style.use('seaborn-paper')

mpl.rcParams['lines.markersize']=10
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=20
mpl.rcParams['ytick.labelsize']=20
mpl.rcParams['axes.labelsize']=20
mpl.rcParams['axes.titlesize']=20
mpl.rcParams['figure.figsize']=(10,10)

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
# Ensemble plots 

d1,d2,d3=37,38,39

t_stop=100
t_start=0
plt.figure(figsize=(10,10))
 
# syntax for 3-D projection
#ax = plt.axes(projection ='3d')
 
plt.plot(a1_ens[t_start:t_stop,d3,:],a_ens[t_start:t_stop,d2,:],marker='.',c='g',markersize=10,alpha=0.4)
plt.plot(a_ens[t_start:t_stop,d3,:],a_ens[t_start:t_stop,d2,:],marker='.',c='b',markersize=10,alpha=0.4)
#plt.scatter(state[0:1000,d1],state[0:1000,d2],marker='.',c='cyan',s=10,alpha=0.5)
#plt.scatter(a_m[:,d1],a_mean[:,d2],marker='.',c='black',s=10,label='mean')
#plt.title('Ensemble on attractor,bias={}'.format(0.0))
plt.xlabel(r'$X\to$')
plt.ylabel(r'$Y\to$')
plt.legend(frameon='True',loc='upper center',fontsize=14)

# Location of plots:
os.chdir('/home/shashank/Dropbox/Data_Assimilation/Thesis_writing/figures')
plt.savefig('stability_on_attractor_bias={}_and_{}.png'.format(5.0,ebias))
plt.show()


# 3d plot

plt.figure(figsize=(10,10))
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
ax.scatter(a_ens[t_start:t_stop,d3,:],a_ens[t_start:t_stop,d2,:],a_ens[t_start:t_stop,d1,:],alpha=0.5)
ax.scatter(a1_ens[t_start:t_stop,d3,:],a1_ens[t_start:t_stop,d2,:],a1_ens[t_start:t_stop,d1,:],c='black',alpha=0.5)

ax.set_xlabel(r'$X \to$')
ax.set_ylabel(r'$Y \to$')
ax.set_zlabel(r'$Z \to$')
plt.legend(frameon='True',loc='upper center',fontsize=14)
plt.savefig('stability_on_attractor_3d_bias={}_and_{}.png'.format(5.0,ebias))
plt.show()