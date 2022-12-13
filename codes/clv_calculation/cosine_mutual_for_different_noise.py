# Calculate the cosines between the true trajectory CLVs and the noisy one
import numpy as np
import os

dt=0.01   
model_dim=3
model_name='L63'
coord=['X','Y','Z']

# making the state noisy
spr=1
t_start=0
t_stop=5000

# Change to data path
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
start_idx=35000 # starting point of the interval of clv( 25000+10000)
base_traj=np.load('State_g={}.npy'.format(dt))[start_idx:start_idx+t_stop]

# CLVs about the true state
base_type='State'
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
 
# calculation of mutual angle between CLVs
cosines_mutual=np.zeros((C.shape[0],3))
for i in range(base_traj.shape[0]):
    for j in range(3):
        if j==2:
            cosines_mutual[i,j]=np.dot(C[i,:,j],C[i,:,-1])
        else:
            cosines_mutual[i,j]=np.dot(C[i,:,j],C[i,:,j+1])

np.save('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/cosines_data/Mutual angle between CLVs_trajectory.npy',cosines_mutual)

for i in range(15):
    mu=(i+1)*1.0
    base_type='state_noisy'
    os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/mu={}'.format(mu))
    C2=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
    G2=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
      
    cosines_mutual=np.zeros((base_traj.shape[0],3))
    for i in range(base_traj.shape[0]):
        for j in range(3):
            if j==2:
                cosines_mutual[i,j]=np.dot(C2[i,:,j],C2[i,:,-1])
            else:
                cosines_mutual[i,j]=np.dot(C2[i,:,j],C2[i,:,j+1])

    np.save('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/cosines_data/Mutual angle between CLVs_mu={}.npy'.format(mu),cosines_mutual)
    print(mu)