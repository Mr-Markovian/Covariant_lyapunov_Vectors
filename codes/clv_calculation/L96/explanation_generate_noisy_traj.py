import numpy as np
import os
import matplotlib.pyplot as plt

model_name='L96'
model_dim=40
dt=0.05
dt_solver=0.01

seed_num=5
# making the state noisy
np.random.seed(seed_num)

# Change to data path
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/explanation'.format(model_dim)
os.chdir(data_path)
true_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf={}_dt_{}_dt_solver={}.npy'.format(dt,600.0,dt,dt_solver))

# We store the noisy trajectory in the new location
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])

#os.mkdir('analysis_sigma=0.0')
os.chdir('analysis_sigma=0.0')
base_type='state_noisy'
np.save('{}_g={}_sigma={}'.format(base_type,dt,0.0),true_traj)
os.chdir('..')

for i in range(5):
    noise_cov=np.load('analysis_cov_ob_sigma={}.npy'.format(sigmas[i]))
    os.mkdir('analysis_sigma={}'.format(sigmas[i]))
    os.chdir('analysis_sigma={}'.format(sigmas[i]))
    noisy_traj=true_traj+np.random.multivariate_normal(np.zeros(model_dim),noise_cov,true_traj.shape[0])
    base_type='state_noisy'
    np.save('{}_g={}_sigma={}'.format(base_type,dt,sigmas[i]),noisy_traj)
    os.chdir('..')
print(noisy_traj.shape)
print('job done')