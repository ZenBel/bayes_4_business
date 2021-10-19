import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats

SUPPLIER_YIELD = np.array([.9, .5, .8])
SUPPLIER_YIELD_SD = np.array([.1, .2, .2])
N_OBS = [30, 20, 2]

def make_yield_data(supplier_yield=SUPPLIER_YIELD,
                    supplier_yield_sd=SUPPLIER_YIELD_SD,
                    n_obs=N_OBS,
                    seed=100
                    ):

    # Generate synthetic data by drawing samples from a beta distribution
    np.random.seed(seed)
    data = []
    for sy, sy_sd, n in zip(supplier_yield, supplier_yield_sd, n_obs):
        data.append(pm.Beta.dist(mu=sy, sd=sy_sd, shape=n).random())

    return data

def make_demand_data():
    return stats.poisson(60, 40).rvs(1000)
