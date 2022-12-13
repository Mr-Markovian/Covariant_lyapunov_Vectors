import numpy as np
import os

#Observation Operator and noise statistics need to be defined first:
mu=0.9        
dim_x=40
m=20   
n=dim_x
ob_gap=0.05
#Observation Operator for observing y 
#Observation operator for alternate grid points:
H2_=np.zeros((m,n))
for i in range(m):
    H2_[i,2*i]=1

obs_cov1=mu*np.eye(m)

trajectory_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-40/assimilated_trajs_seed_3'
os.chdir(trajectory_path)
#Load the trajectory 
State=np.load('state/Multiple_trajectories_N=1_gap=0.05_ti=0.0_tf=600.0_dt_0.05_dt_solver=0.01.npy')

for i in range(5):
    os.chdir(trajectory_path)
    #os.mkdir('ob{}'.format(i+1))
    os.chdir(trajectory_path+'/ob{}'.format(i+1))
    np.random.seed(41+i)
    obs=(H2_@(State.T)).T+np.random.multivariate_normal(np.zeros(m),obs_cov1,State.shape[0])
    np.save('ob{}_gap_{}_H2_'.format(i+1,round(ob_gap,2))+'_mu={}'.format(mu)+'_obs_cov1.npy',obs)

print(State.shape)
print('Job Done')

