import numpy as np
import os

#seed for observation ob1 was at 45,subsequent obs with +1 seed.
# Code for shortening the object in time :
def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

#Observation Operator and noise statistics need to be defined first:
mu=1.0           #change this
dim_x=3
m1=1   
n1=dim_x
#Observation Operator for observing y 
H1_=np.zeros((m1,n1))
H1_[0,m1]=1          # for y, choose the location for
obs_cov1=mu*np.eye(m1)

#Observation Operator for observing x and z 
#H1_=np.zeros((m1,n1))
#H1_[:,1:]=np.eye(m1)  
#obs_cov1=mu*np.eye(m1)

obs_gap=0.01
factor_=int(obs_gap/0.01)       
assims_=int(50000/factor_)

os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
#Load the trajectory 
State=shorten(np.load('Trajectory_{}_T={}.npy'.format(0.01,500)),factor=factor_)

str1_=os.getcwd()
for i in range(5):
    os.chdir(str1_)
    #os.mkdir('ob{}'.format(i+1))
    os.chdir(str1_+'/ob{}'.format(i+1))
    np.random.seed(41+i)
    obs=(H1_@(State.T)).T+np.random.multivariate_normal(np.zeros(m1),obs_cov1,assims_)
    np.save('ob{}_gap_{}_H1_'.format(i+1,round(obs_gap,2))+'_mu={}'.format(mu)+'_obs_cov1.npy',obs)

print('Job Done')
