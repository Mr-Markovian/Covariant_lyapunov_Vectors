import jax.numpy as jnp
import numpy as np

# L96 2-level model, 
def L96_2level(t,x_,d_x,d_y,F_x,c=10,b=10,h=1):
    "2 scale lorenz 96 model, the input is combined vector x_=[x y] "
    X=x_[0:d_x]
    Y=x_[d_x:].reshape(d_x,d_y)
    dx_dt=-jnp.roll(X,-1)*(jnp.roll(X,-2)-jnp.roll(X,1))-X+F_x-(h*c/b)*jnp.sum(Y,axis=1)
    dy_dt=-c*b*(jnp.roll(Y,1,axis=1)*(jnp.roll(Y,2,axis=1)-jnp.roll(Y,-1,axis=1)))-c*Y+(h*c/b)*jnp.tile(X,(d_y,1)).T
    dx=jnp.zeros_like(x_)
    dx[0:d_x]=dx_dt
    dx[d_x:]=dy_dt.flatten()
    return dx

# L96 single level model with forcing=10
def L96(x_,forcing=8.):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(jnp.roll(x_,-1)-jnp.roll(x_,2))*jnp.roll(x_,1)-x_+forcing
    return dx_dt

#Lorenz-63 with default values: sigma=10, rho=28, beta=8/3
def L63(x,sigma=10.,rho=28.,beta=8./3):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    return jnp.array([sigma*(x[1]-x[0]),x[0]*(rho-x[2])-x[1],x[0]*x[1]-beta*x[2]])
    

