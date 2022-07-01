# Lyapunov Exponents from system of ODE
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp 
import json
from jax import partial,jvp,vmap,jit 
import jax.numpy as jnp
np.random.seed(35)
"""
# Linearized matrix :
def tangent_evolve(x_):
    x1=x_[:,0]  # Nx1
    x2=x_[:,1:] # NxN
    D_A=np.zeros((N,N))
    D_A[N-1,0],D_A[N-1,N-1]=x1[N-2],-1
    D_A[N-1,N-2],D_A[N-1,N-3]=x1[0]-x1[N-3],-x1[N-2]
    for i in range(N-1):
        D_A[i,i+1],D_A[i,i]=x1[i-1],-1
        D_A[i,i-1],D_A[i,i-2]=x1[i+1]-x1[i-2],-x1[i-1]
    return np.dot(D_A,x2)

def nonlinear(x_):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(np.roll(x_,1)-np.roll(x_,-2))*np.roll(x_,-1)-x_
    return np.transpose([dx_dt+forcing])

# The def function is the above linearized matrix multiplied by the vector 
def fun(t,x1):
    x_b=x1.reshape((N,N+1))
    #Take the vector,create a matrix, do calculations and convert back into a vector
    c1=nonlinear(x_b[:,0])
    c2=tangent_evolve(x_b)
    rhs=np.concatenate((c1,c2),axis=1)	
    return rhs.flatten()
"""

def fun(t,x):
    "RHS for scipy.integrate.solve_ivp"
    return np.asarray(system_plus_jacobian(x))

@jit
def system_plus_jacobian(x_):
    "Linearized dynamics by computing the jacobian at a point"
    X_=x_.reshape((N+1,N))
    x_eval=X_[0]  # Nx1
    in_tangents=X_[1:,:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, L96_dynamics , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(0), out_axes=(None,0))((in_tangents,)) 
    return jnp.concatenate((y,out_tangents.ravel()))


@jit
def L96_dynamics(x_):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(jnp.roll(x_,-1)-jnp.roll(x_,2))*jnp.roll(x_,1)-x_
    dx_dt=dx_dt+forcing   # the rhs of the system
    return jnp.transpose(dx_dt)

#Modified Gram Schmid Orthogonalization
@jit 
def modified_gram_schmidt(M):
    Q=np.zeros_like(M)
    for i in range(N):
        q=M[:,i]/norm(M[:,i])
        for j in range(i+1,N):
                M[:,j]=M[:,j]-np.dot(np.transpose(q),M[:,j])*q
        Q[:,i]=q
    return Q

delta_t=0.01
t_f=100
T=np.arange(0,t_f,delta_t)

# Running over many initial conditions for computing exponents:
""" Choice of initial vectors can be standard orthonormal vectors """
with open('L_96_I_x0_on_attractor_dim_x_40,forcing_8.json'.format(35)) as jsonfile:
     param=json.load(jsonfile)

# Number of variables in the problem 
N=param['dim_x']
forcing=param['forcing_x']
# Initial condition for the whole system is ready below
Exponents=np.zeros((N,len(T)))
# Initial condition for the perturbations:
x_p=np.eye(N)
x_o=np.transpose([np.asarray(param['initial'])[:N]])
x_init=np.concatenate((x_o,x_p),axis=1)
X_o=x_init.flatten()

# Now run the above to get all the exponents
 
for j in range(len(T)):
    solution=solve_ivp(fun,(T[j],T[j]+delta_t),X_o)
    X_f=solution.y[:,-1].reshape(N,N+1)
    O_p=X_f[:,1:]
    for i in range(N):
        Exponents[i,j]=np.log(norm(O_p[:,i]))/delta_t
    O=np.concatenate((np.transpose([X_f[:,0]]),modified_gram_schmidt(O_p)),axis=1)
    X_o=O.flatten()

np.save('exponents_vers_t_seed{},dim={},f={}.npy'.format(35,N,forcing),np.mean(Exponents,axis=1))
#plt.figure(figsize=(16,8))
#plt.scatter(np.arange(N),np.mean(Exponents,axis=1),marker='o',color='navy',markeredgecolor='black',label='$\delta t$={}'.format(delta_t))
#plt.xlabel('i',fontsize=18)
#plt.ylabel('$\lambda_i$',fontsize=18)
#plt.xticks(np.arange(0,N))
#plt.title('Lyapunov spectrum for {}-dimensional Lorenz-96 system',fontsize=18)
#plt.legend()
#plt.savefig('Lyapunov_exponents_seed{}_t={}_dim={},f={}.png'.format(35,t_f,N,forcing))
#plt.show()  
print('Job DOne')




