"""This is the code which finally runs the Assimilation for a desired set of
parameters which are bias and Covariance of the initial ensemble.
We run the experiment for 10 observation realisation."""

from Enkf_ import *
#Creating a diagonal matrix for the observation of first 10 variables
import numpy as np
import json
import os

#------------------------------Function defined----------------------------------
# Code for shortening the object in time :
def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

# Code for obtaining partial observations
def partial(arr,ax_=1):
    # trim the last ax_
    if ax_==0:
        return arr
    else:
        arr_shape=list(arr.shape)
        arr_shape[1]=arr_shape[1]-ax_
        new_arr=np.zeros(arr_shape)
        new_arr=arr[:,:-ax_]
        return new_arr
#-------------------------------------------------------------------------------

#Loading the default parameters from a json file
with open('Lorenz_63_assim_parameters.json') as jsonfile:
    parameters=json.load(jsonfile)

#The model and observation covariance matrix are both diagonal.
#Order of magnitude of matrix elements in the matrices for Model and observation covariance matirx.

# Parameters to be controlled from here...
parameters['observables']=1
m1        =parameters['observables']       # Number of observed Components
n1        =parameters['dim']               # Number of grid point state
parameters['obs_gap']=0.01
#lambda_   =parameters['lambda']            #Magnitude of diagonal elements of Initial Ensemble Covariance
parameters['mu']=1.0
mu        =parameters['mu']                #Magnitude of diagonal elements of Observation Covariance
parameters['N'] =40
N         =parameters['N']
obs_gap   =parameters['obs_gap']
parameters['alpha']      =1.1              #The inflation factor
parameters['loc']='False'
parameters['loc_fun']='none'
parameters['l_scale']=0

#Number of Assimilation to perform:
parameters['assimilations']=50000

#Observation Operator for observing y 
H1_=np.zeros((m1,n1))
H1_[0,1]=1     # for y, choose the location for

#Creating a diagonal observation covariance matirx
obs_cov1=mu*np.eye(m1)

#initial bias in the ensemble........
biases=np.array([6.0])
covs_=np.array([4.0])

#------------------------Go to the desired folder-------------------------------
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
str1_=os.getcwd()

#iterate over observations:
for w in range(1):
    os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))
    parameters['obs']=np.load('ob{}_gap_{}_H1_'.format(w+1,obs_gap)+'_mu={}'.format(mu)+'_obs_cov1.npy')
    # iterate over the ensembles
    for i in range(len(biases)):
    #bias in mean.This choice is implemented through the initial ensemble.
        bias=biases[i]
        for j in range(len(covs_)):
            #setting the seed for reproducibility
            seed_num=int(47*(i+1)*(j+1))
            np.random.seed(seed_num)
            parameters['lambda']=covs_[j]
            Initial_cov1=parameters['lambda']*np.eye(parameters['dim'])
            parameters['model_cov']=Initial_cov1  # model error covariance matrix
            parameters['obs_cov']  =obs_cov1      # Observation error covariance matrix
            parameters['H']        =H1_
            # Load the ensembles from where they are saved:
            str_2=os.path.join(str1_,'ensembles')
            Ens=np.load(str_2+'/Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(50,bias,covs_[j],47))
            parameters['initial_ensemble']=Ens[:,:parameters['N']]
            parameters['initial']=np.mean(Ens[:,:parameters['N']],axis=1)
            #print(os.getcwd())
            #parameters['obs']=partial(np.load('ob{}_gap_{}_H1_'.format(w+1,0.2)+'_mu={}'.format(mu)+'_obs_cov1.npy'),n1-m1)
            os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))
            obj=Enkf(parameters)
            obj.simulate()
            obj.label='bias={}_'.format(bias)+obj.label
            os.mkdir(obj.label)
            os.chdir(os.getcwd()+'/'+obj.label)
            obj.save_data()
            print('ob={},bias={},cov={}-run completed'.format(w+1,biases[i],covs_[j]))
            os.chdir(os.path.join(str1_,'ob{}'.format(w+1)))

#Setting parameters for Observation matrix,observation error-covariance and model error-covariance
# Creating an object with the above parameters specification
print('Job Done')
