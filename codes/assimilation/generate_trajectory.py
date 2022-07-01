from lorenz_63_ode import *
#from II_scale_lorenz_96 import *
import numpy as np
import json
#seed_num=31
import os

"""Loading parameters and initial condition for lorenz_96"""
parametersFile='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/codes/assimilation/L_63_x0_seed_45.json'
with open(parametersFile) as jsonfile:
     parameters=json.load(jsonfile)

#Total time of the trajectory
parameters['time_start']=0
parameters['time_stop']=500
parameters['obs_gap']=0.01

#integrate to generate the trajectory:
parameters['t_evaluate']=np.arange(0,parameters['time_stop'],parameters['obs_gap'])
ob_=lorenz_63(parameters)
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
np.save('Trajectory_{}_T={}.npy'.format(parameters['obs_gap'],parameters['time_stop']),ob_.solution())
print('Job Done')
