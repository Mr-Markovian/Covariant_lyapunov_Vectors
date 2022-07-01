"""Definig my own rk4 solver module. This contains rk4 which can work for a single trajectory,
and an rk4 which works for an ensemble of trajectories"""

def rk4_solver(rhs_function,x_initial,time_step):
    "Returns x_k+1 from x_k via rk4 scheme"
    x_o=x_initial
    x1= rhs_function(x_initial)*time_step
    x2= rhs_function(x_initial+x1/2.0)*time_step
    x3= rhs_function(x_initial+x2/2.0)*time_step
    x4= rhs_function(x_initial+x3)*time_step
    return x_o+(x1 + 2*x2 + 2*x3 + x4)/6

#def vmap_rk4():
#    "Returns a batch of x_k+1 for a batch of x_k using rk4 scheme"
#    vmap(rk4,)

#def edtrk4(rhs_function,x_initial,parameters,dt):
