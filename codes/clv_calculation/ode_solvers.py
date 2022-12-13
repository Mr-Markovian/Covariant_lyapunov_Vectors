"""Definig my own rk4 solver module. This contains rk4 which can work for a single trajectory,
and an rk4 which works for an ensemble of trajectories.
It also has a stiff-pde solver EDTRK4 developed by Cox and Matthews.
S.M. Cox; P.C. Matthews (2002). Exponential Time Differencing for Stiff Systems. , 176(2), 430–455. doi:10.1006/jcph.2002.6995 
"""
import jax.numpy as jnp

def rk4_solver(rhs_function,x_initial,time_step):
    "Returns x_k+1 from x_k via rk4 scheme"
    x_o=x_initial
    k1= rhs_function(x_initial)
    k2= rhs_function(x_initial+k1*time_step/2.0)
    k3= rhs_function(x_initial+k2*time_step/2.0)
    k4= rhs_function(x_initial+k3*time_step)
    return x_o+(k1 + 2*k2 + 2*k3 + k4)*time_step/6.0

def EDTRK4_solver_single(rhs_function_lin,rhs_function_nonlin,k,x_initial,time_step):
    u_n=x_initial
    h=time_step
    IF=jnp.exp(rhs_function_lin*time_step)
    IF_by_2=jnp.exp(rhs_function_lin*time_step/2)
    a_n =u_n@IF_by_2+(IF_by_2-1)@rhs_function_nonlin(u_n)
    b_n =u_n@IF_by_2+(IF_by_2-1)@rhs_function_nonlin(a_n)
    c_n =a_n@IF_by_2+(IF_by_2-1)@(2*rhs_function_nonlin(b_n)-rhs_function_nonlin(u_n))
    part_1=(-4-h*k+IF*(4-3*k*h+(h**2)*(k**2)))
    part_2=2+h*k+IF*(-2+h*k)
    part_3=-4-3*k*h-(k**2)*(h**2)+IF*(4-h*k)
    F_un=rhs_function_nonlin(u_n,h/2)
    F_an=rhs_function_nonlin(a_n,h/2)
    F_bn=rhs_function_nonlin(b_n,h/2)
    F_cn=rhs_function_nonlin(c_n,h/2)
    u_n_plus_1=u_n@IF+(F_un*part_1)+2*(F_an+F_bn)*part_2+F_cn*part_3
    return u_n_plus_1

# def EDTRK4_solver(rhs_function_lin,rhs_function_nonlin,k,x_initial,time_step):
#     u_n=x_initial
#     h=time_step
#     IF=jnp.diag(jnp.exp(rhs_function_lin*h))
#     IF_by_2=jnp.diag(jnp.exp(rhs_function_lin*h/2))
#     a_n=IF_by_2@u_n+(IF_by_2-1)@rhs_function_nonlin(u_n)
#     b_n=IF_by_2@u_n+(IF_by_2-1)@rhs_function_nonlin(a_n)
#     c_n=IF_by_2@a_n+(IF_by_2-1)@(2*rhs_function_nonlin(b_n)-rhs_function_nonlin(u_n))
    


