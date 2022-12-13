import numpy as np
from scipy.integrate import solve_ivp  # to be replaced by solve_ivp later

def fun(t,x,forcing_):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(np.roll(x,-1)-np.roll(x,2))*np.roll(x,1)-x
    return dx_dt+forcing_

class lorenz_96:
    """create a class to solve lorenz 96 system.create objects with attributes
    variables-number of variables  ;forcing- the forcing constant ;
    time_start= starting time of solution ; time_stop=ending time of the solution initial
    ,the initial value of the system at t start."""

    def __init__(self,parameters):
        "To intialize the variables "
        # parse parameters from the dictionary
        for key in parameters:
            setattr(self, key, parameters[key])
        #self.time_units=int((self.time_stop-self.time_start)/self.time_step)
        #self.t_evaluate=None

    def solution(self):
        "To generate the full solution from starting time to stop time"
        flow=solve_ivp(fun,[self.time_start,self.time_stop],self.initial,method='RK45',t_eval=self.t_evaluate,args=(self.forcing_x,))
        self.soln=(flow.y).T
        #self.time=flow.t
        return self.soln
