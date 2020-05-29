import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIHRModel
from .util import observe, ExponentialRandomWalk

import numpy as onp

from covid.models.base import Model

"""
************************************************************
SEIHR model
************************************************************
"""


class SEIHRBase(Model):

    compartments = ['S', 'E', 'I', 'LH', 'SH', 'R']

    @property
    def obs(self):
        '''Provide extra arguments for observations
        
        Used during inference and forecasting
        '''        
        if self.data is None:
            return {}

        return {
            'confirmed': self.data['confirmed'].values,
            'hosp': self.data['hosp'].values
           }
    
    
    def dz_mean(self, samples, **args):
        '''Daily hosps mean'''
        mean_z = self.mean_z(samples, **args)
        if args.get('forecast'):
            first = self.mean_z(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan
            
        return onp.diff(mean_z, axis=1, prepend=first)        
    
    def dz(self, samples, noise_scale=0.4, **args):
        '''Daily hosps with observation noise'''
        dz_mean = self.dz_mean(samples, **args)
        dz = dist.Normal(dz_mean, noise_scale * dz_mean).sample(PRNGKey(10))
        return dz
        
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
                 alpha_shape = 8,
                 sigma_shape = 8,
                 gamma_shape = 8,
                 det_prob_est = 0.15,
                 det_prob_conc = 50,
                 det_noise_scale = 0.15,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0,
                 drift_scale = None,
                 confirmed=None,
                 hosp=None):

        '''
        Stochastic SEIHR model. Draws random parameters and runs dynamics.
        '''        
                
        # Sample initial number of infected individuals

        I0 = numpyro.sample("I0", dist.Uniform(10, 0.05*N))

        E0 = numpyro.sample("E0", dist.Uniform(10, 0.05*N))

        R0 = numpyro.sample("R0", dist.Uniform(0, 0.01*N))

        LH0 = numpyro.sample("LH0", dist.Uniform(0, 0.01*N))

        SH0 = numpyro.sample("SH0", dist.Uniform(0, 0.01*N))

        # Sample parameters
        alpha = numpyro.sample("alpha", 
                               dist.Gamma(alpha_shape, alpha_shape * E_duration_est))

        sigma = numpyro.sample("sigma", 
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))

        gamma = numpyro.sample("gamma", 
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

    #     gamma = numpyro.sample("gamma", 
    #                            dist.TruncatedNormal(loc = 1./I_duration_est, scale = 0.25)

        beta0 = numpyro.sample("beta0", 
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

        det_prob = numpyro.sample("det_prob", 
                                  dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))

        lhosp_prob = numpyro.sample("lhosp_prob", 
                                    dist.Beta(.1 * 100,
                                              (1-.1) * 100))

        lhosp_rate = numpyro.sample("lhosp_rate", 
                                    dist.Gamma(10, 10 * 10))

        shosp_prob = numpyro.sample("shosp_prob", 
                                    dist.Beta(.1 * 100,
                                              (1-.1) * 100))

        shosp_rate = numpyro.sample("shosp_rate", 
                                    dist.Gamma(10, 10 * 10))

        if drift_scale is not None:
            drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
        else:
            drift = 0


        x0 = SEIHRModel.seed(N=N, I=I0, E=E0, R=R0, LH=LH0, SH=SH0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        confirmed0, confirmed = (None, None) if confirmed is None else (confirmed[0], confirmed[1:])
        hosp0, hosp = (None, None) if hosp is None else (hosp[0], hosp[1:])


        # First observation
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = observe("y0", x0[6], det_prob, det_noise_scale, obs=confirmed0)
            
        with numpyro.handlers.scale(scale_factor=2.0):
            z0 = observe("z0", x0[5], .99, det_noise_scale, obs=hosp0)

        params = (beta0, sigma, gamma, 
                  rw_scale, drift, 
                  det_prob, det_noise_scale, 
                  lhosp_prob, lhosp_rate, shosp_prob, shosp_rate)

        beta, x, y, z = self.dynamics(T, params, x0, confirmed=confirmed, hosp=hosp)

        x = np.vstack((x0, x))
        y = np.append(y0, y)
        z = np.append(z0, z)

        if T_future > 0:

            params = (beta[-1], sigma, gamma, 
                      forecast_rw_scale, drift, 
                      det_prob, det_noise_scale, 
                      lhosp_prob, lhosp_rate, shosp_prob, shosp_rate)

            beta_f, x_f, y_f, z_f = self.dynamics(T_future+1, params, x[-1,:], 
                                                  suffix="_future")

            x = np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)

        return beta, x, y, z, det_prob, (lhosp_prob + shosp_prob)

    
    def dynamics(self, T, params, x0, confirmed=None, hosp=None, suffix=""):
        '''Run SEIRD dynamics for T time steps'''

        beta0, sigma, gamma, rw_scale, drift, \
        det_prob, det_noise_scale, lhosp_prob, lhosp_rate, shosp_prob, shosp_rate  = params

        beta = numpyro.sample("beta" + suffix,
                      ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

        # Run ODE
        x = SEIHRModel.run(T, x0, (beta, sigma, gamma, lhosp_prob, lhosp_rate, shosp_prob, shosp_rate))
        x = x[1:] # first entry duplicates x0
        numpyro.deterministic("x" + suffix, x)


        # Noisy observations
        with numpyro.handlers.scale(scale_factor=0.5):
            y = observe("y" + suffix, x[:,6], det_prob, det_noise_scale, obs = confirmed)

        with numpyro.handlers.scale(scale_factor=2.0):
            z = observe("z" + suffix, x[:,5], .99, det_noise_scale, obs = hosp)

        return beta, x, y, z
        