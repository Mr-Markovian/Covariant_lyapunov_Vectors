import numpy as np
import os

model_name='L63'
model_dim=3
dt=0.01
dt_solver=0.005

seed_num=3
# making the state noisy
np.random.seed(seed_num)

# Change to data path
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63/seed_{}'.format(model_dim,seed_num)
os.chdir(data_path)
true_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf={}_dt_{}_dt_solver={}.npy'.format(dt,300.0,dt,dt_solver))

# We store the noisy trajectory in the new location
#sigmas=np.array([0.01,0.03,0.05])
# new sigmas
sigmas=np.array([6.4])
#sigmas=np.array([0.45,0.5,0.52,0.55])

#os.mkdir('sigma=0.0')
#os.chdir('sigma=0.0')
#base_type='state_noisy'
#np.save('{}_g={}_sigma={}'.format(base_type,dt,0.0),true_traj)
#os.chdir('..')

for i in range(1):
    noise_cov=sigmas[i]**2*np.eye(model_dim)
    #if (os.path.exists('sigma={}'.format(sigmas[i]))=='False'):
    os.mkdir('sigma={}'.format(sigmas[i]))
    os.chdir('sigma={}'.format(sigmas[i]))
    noisy_traj=true_traj+np.random.multivariate_normal(np.zeros(model_dim),noise_cov,true_traj.shape[0])
    base_type='state_noisy'
    np.save('{}_g={}_sigma={}'.format(base_type,dt,sigmas[i]),noisy_traj)
    os.chdir('..')
print(noisy_traj.shape)
print('job done')