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

V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  

for i in range(6,15):
    mu=(i+1)*1.0
    base_type='state_noisy'
    os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/mu={}'.format(mu))
    C2=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
    G2=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
    #noisy_traj=np.load('{}_g={}_mu={}.npy'.format(base_type,dt,mu))[10000+t_start:10000+t_stop]
    V2=np.zeros_like(G)
    for i in range(G.shape[0]):
        V2[i]=G2[i]@C2[i]    

    cosines=np.zeros((base_traj.shape[0],3))
    for i in range(base_traj.shape[0]):
        for j in range(3):
            cosines[i,j]=np.absolute(np.dot(V[i,:,j],V2[i,:,j]))

    np.save('cosines_rel_mu={}.npy'.format(mu),cosines)
    print(mu)