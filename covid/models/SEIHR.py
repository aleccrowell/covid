import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIHRModel
from .util import observe, observe_nonrandom, ExponentialRandomWalk

import numpy as onp

from covid.models.base import Model

"""
************************************************************
SEIHR model
************************************************************
"""


class SEIHRBase(Model):

    compartments = ['S', 'E', 'I', 'H', 'R']

    @property
    def obs(self):
        '''Provide extra arguments for observations
        
        Used during inference and forecasting
        '''        
        if self.data is None:
            return {}

        return {
            #'confirmed': self.data['confirmed'].values,
            'hosp': self.data['hosp'].values
           }

    def dy_mean(self, samples, **args):
        '''Daily confirmed cases mean'''
        mean_y = self.mean_y(samples, **args)
        if args.get('forecast'):
            first = self.mean_y(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan
            
        return onp.diff(mean_y, axis=1, prepend=first)
    
    def dy(self, samples, noise_scale=0.4, **args):
        '''Daily confirmed cases with observation noise'''
        dy_mean = self.dy_mean(samples, **args)
        dy = dist.Normal(dy_mean, noise_scale * dy_mean).sample(PRNGKey(11))
        return dy
        


class SEIHR(SEIHRBase):    
    
    def __call__(self,
                 T = 50,
                 N = 1e5,
                 T_future = 0,
                 E_duration_est = 5.5,
                 I_duration_est = 3.0,
                 R0_est = 3.0,
                 beta_shape = 1,
                 sigma_shape = .2,
                 gamma_shape = .2,
                 det_prob_est = 0.15,
                 det_prob_conc = 50,
                 det_noise_scale = 0.15,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0,
                 drift_scale = None,
                 #confirmed=None,
                 hosp=None):

        '''
        Stochastic SEIHR model. Draws random parameters and runs dynamics.
        '''        
                
        # Sample initial number of infected individuals

        I0 = numpyro.sample("I0", dist.Uniform(10, 0.05*N))

        E0 = numpyro.sample("E0", dist.Uniform(10, 0.05*N))

        R0 = numpyro.sample("R0", dist.Uniform(0, 0.01*N))

        H0 = numpyro.sample("H0", dist.Uniform(0, 0.01*N))

        # Sample parameters

        sigma = numpyro.sample("sigma", 
                               dist.LogNormal(np.log(1/E_duration_est),sigma_shape))

        gamma = numpyro.sample("gamma", 
                                dist.LogNormal(np.log(1/I_duration_est),gamma_shape))

    #     gamma = numpyro.sample("gamma", 
    #                            dist.TruncatedNormal(loc = 1./I_duration_est, scale = 0.25)

        beta0 = numpyro.sample("beta0", 
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

        det_prob = numpyro.sample("det_prob", 
                                  dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))

        hosp_prob = numpyro.sample("hosp_prob", 
                                    dist.Beta(.1 * 100,
                                              (1-.1) * 100))

        if drift_scale is not None:
            drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
        else:
            drift = 0


        x0 = SEIHRModel.seed(N=N, I=I0, E=E0, R=R0, H=H0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        #confirmed0, confirmed = (None, None) if confirmed is None else (confirmed[0], confirmed[1:])
        hosp0, hosp = (None, None) if hosp is None else (hosp[0], hosp[1:])


        # First observation
        with numpyro.handlers.scale(scale_factor=2.0):
            y0 = observe_nonrandom("y0", x0[3], det_noise_scale, obs=hosp0)

        params = (beta0, sigma, gamma, 
                  rw_scale, drift, 
                  det_prob, det_noise_scale, 
                  hosp_prob)

        beta, x, y = self.dynamics(T, params, x0, hosp=hosp)

        x = np.vstack((x0, x))
        y = np.append(y0, y)
        #z = np.append(z0, z)

        if T_future > 0:

            params = (beta[-1], sigma, gamma, 
                      forecast_rw_scale, drift, 
                      det_prob, det_noise_scale, 
                      hosp_prob)

            beta_f, x_f, y_f = self.dynamics(T_future+1, params, x[-1,:], 
                                                  suffix="_future")

            x = np.vstack((x, x_f))
            y = np.append(y, y_f)
            #z = np.append(z, z_f)

        return beta, x, y, det_prob, hosp_prob

    
    def dynamics(self, T, params, x0, hosp=None, suffix=""):
        '''Run SEIRD dynamics for T time steps'''

        beta0, sigma, gamma, rw_scale, drift, \
        det_prob, det_noise_scale, hosp_prob  = params

        beta = numpyro.sample("beta" + suffix,
                      ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

        # Run ODE
        x = SEIHRModel.run(T, x0, (beta, sigma, gamma, hosp_prob))
        x = x[1:] # first entry duplicates x0
        numpyro.deterministic("x" + suffix, x)


        # Noisy observations
        with numpyro.handlers.scale(scale_factor=2.0):
            y = observe_nonrandom("y" + suffix, x[:,3], det_noise_scale, obs=hosp)

        return beta, x, y, z
        