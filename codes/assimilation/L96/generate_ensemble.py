import numpy as np
import os
import json

"""Loading parameters and initial condition for lorenz_63"""
trajectory_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-40/assimilated_trajs_seed_3'
os.chdir(trajectory_path)
#Load the trajectory 
State=np.load('state/Multiple_trajectories_N=1_gap=0.05_ti=0.0_tf=600.0_dt_0.05_dt_solver=0.01.npy')

#initial bias in the ensemble........
biases=np.array([-5.0])

#initial ensemble spread
covs_=np.array([1.0])
#Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])

# Finalized experiments in April,2021
#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
#os.mkdir('ensembles')
os.chdir(os.path.join(os.getcwd(),'ensembles'))
for i in range(len(biases)):
    bias=biases[i]
    for j in range(len(covs_)):
        "Select seeds such that there is a unique value for each set of complete parameters"
        seed_num=int(47*(i+1)*(j+1))
        np.random.seed(seed_num)
        mu=covs_[j]
        Initial_cov=mu*np.eye(40)
        x0_ensemble=np.random.multivariate_normal(np.asarray(State[0])+bias,Initial_cov,50).T
        np.save('Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(50,bias,covs_[j],seed_num),x0_ensemble)
        #np.save('Ensemble={}_init_cov_{}_seed_{}.npy'.format(i,parameters['lambda_'],seed_num),x0_ensemble)

print('Job Done')

