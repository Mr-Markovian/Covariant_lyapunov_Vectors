import numpy as np
import os
import matplotlib.pyplot as plt

model_name='L63'
model_dim=3
dt=0.01

# making the state noisy
np.random.seed(2111997)
mu=81.0
noise_cov=mu*np.eye(model_dim)

# Change to data path
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state')
start_idx=25000 # starting point of the forward transient
true_traj=np.load('State/State_g={}.npy'.format(dt))[start_idx:]

# We store the noisy trajectory in the new location
os.mkdir('mu={}'.format(mu))
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/mu={}'.format(mu))

noisy_traj=true_traj+np.random.multivariate_normal(np.zeros(model_dim),noise_cov,true_traj.shape[0])
base_type='state_noisy'
np.save('{}_g={}_mu={}'.format(base_type,dt,mu),noisy_traj)
plt.figure()
plt.plot(np.arange(noisy_traj.shape[0]),noisy_traj[:,1])
plt.plot(np.arange(noisy_traj.shape[0]),true_traj[:,1])
plt.show()
print('job done')