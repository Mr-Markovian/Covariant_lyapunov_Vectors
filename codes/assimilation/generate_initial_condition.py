
"""The Set of codes here generate an initial condition on the attractor for the desired lorenz_96 of given dimension
by integrating from some arbitrary initial condition and integrating"""

from lorenz_96_ode import *
#from II_scale_lorenz_96 import *
import numpy as np
import json
import os
import random
random.seed(35)

#Create a dictionary of parameters
parameters={}

#For generating initial condition and trajectory for the 2 level lorenz system.

dim_=10
# Fill in the parameters for the desired system
parameters={}
parameters['dim_x']=dim_
parameters['initial']=10*np.random.rand(dim_)
parameters['time_start']=0
parameters['time_stop']=5000
#parameters['time_step']=0.1
parameters['forcing_x']=10
parameters['t_evaluate']=None

ob2=lorenz_96(parameters)
ob2.solution()
print((ob2.soln)[-1].shape)
parameters['initial']=(ob2.soln)[-1].tolist()
os.chdir('..')
parametersFile='L_96_I_x0_on_attractor_dim_{},forcing_{}.json'.format(dim_,parameters['forcing_x'])
with open(parametersFile, "w") as jsonFile:
        json.dump(parameters, jsonFile, indent=4)

#Generating initial condition and trajectory for lorenz_96_II
"""
dim_x,dim_y=20,10
parameters['dim_x']=dim_x
parameters['dim_y']=dim_y
parameters['initial']=5*np.random.rand(dim_x*(dim_y+1))
parameters['forcing_x']=15
#parameters['forcing_y']=
parameters['c']=10
parameters['b']=10
parameters['h']=1
parameters['time_start']=0
parameters['time_stop']=1000
parameters['t_evaluate']=None

ob2=lorenz_96_II(parameters)
print(ob2.solution()[-1])
parameters['initial']=ob2.solution()[-1].tolist()

parametersFile='lorenz_96_II_x0_dim_x_y_{},{},forcing_{}.json'.format(dim_x,dim_y,parameters['forcing_x'])
with open(parametersFile, "w") as jsonFile:
        json.dump(parameters, jsonFile, indent=4)
"""
