'''
Geometric Brownian Motion (GBM)
'''

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GBMParameters:
    mu: float
    sigma: float


def gbm_log_return_mean(gbm_params: GBMParameters, dt: float) -> float:
    return (gbm_params.mu - gbm_params.sigma**2 / 2) * dt


def gbm_log_return_std(gbm_params: GBMParameters, dt: float) -> float:
    return gbm_params.sigma * math.sqrt(dt)


def log_returns(
        gbm_params: GBMParameters, 
        n: int, 
        dt: float, 
        rng: np.random.Generator
) -> np.ndarray:
    '''
    generates a total count of n i.i.d log return:
    ln(P_1/P_0) = ln(P_1) - ln(P_0)
                = (mu - sigma**2 / 2) * dt + sigma * sqrt(dt) * epsilon
    where
        1. don't forgot the drift corection term
        2. epsilon ~ N(0, 1)
    '''
    return rng.normal(
        loc=gbm_log_return_mean(gbm_params, dt),
        scale=gbm_log_return_std(gbm_params, dt),
        size=n,
    )