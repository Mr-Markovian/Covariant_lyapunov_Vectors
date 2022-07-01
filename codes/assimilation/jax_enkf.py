"""Jaxed version of enkf"""

# can one create a filter object on which we can then run the jax part.
# All the assimilation setup make a filter object with all the details as attributed.

@jit
def forecast(last_analysis_ensemble):
    
    return forecast_ensmble

@jit
def analysis(forecast_ensemble):

    return analysis_ensemble

my_solver=partial(solver,ode_rhs=propagator)


