# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:23:11 2022

@author: de_hauk
"""


import numpy as np
import pickle
import pandas as pd
import os
import os
from collections import defaultdict
import torch
import numpy as np
import scipy.stats
from torch.distributions import constraints
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
from pyro import optim, poutine
from pyro.distributions import constraints
from pyro.distributions.transforms import block_autoregressive, iterated
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormalizingFlow
from pyro.infer.reparam import NeuTraReparam
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import pyreadr

os.chdir(r'C:\Users\de_hauk\Documents\GitHub\mort_model\New folder')
file_pop = r'maple.population.rda'
file_death = r'maple.deaths.rda'
df_pop = pyreadr.read_r(file_pop)
df_death = pyreadr.read_r(file_death)


#### sample data from here
N = 100
x_var = np.random.normal(loc = 6., 
                         scale = 2., 
                         size = N)

y_var = np.random.normal(loc = x_var, 
                           scale = 1.)
def run_hmc(args, model):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, warmup_steps=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(args.param_a, args.param_b)
    mcmc.summary()
    return mcmc

data_curr = (x_var,y_var)


def model_INLA(x,y):
    # Global variables.
    alpha = pyro.sample('alpha',
                        dist.Normal(0, 0.001))
    
    beta  = pyro.sample('beta',
                        dist.Normal(0, 0.001))
    
    tau   = pyro.sample('Gamma',
                        dist.Gamma(0.01,0.01))
    
    coefs_mean = torch.zeros(X_train_torch.size())
    for i in torch.arange(X_train_torch.size()):
        mu = alpha + beta * x[i]
        obs =  pyro.sample('obs', 
                          dist.Normal(mu, tau), 
                          obs = y[i])

X_train_torch = torch.tensor(x_var)
y_train_torch = torch.tensor(y_var)
# Clear the parameter storage
pyro.clear_param_store()

nuts_kernel = NUTS(model_INLA)

sampler = MCMC(nuts_kernel, 
               warmup_steps = 500, 
               num_samples = 1000)
sampler.run(X_train_torch,y_train_torch)

# mcmc.summary()
