# Generate simulated prices

import numpy as np
import genesis.gbm as gbm

from typing import Tuple
from dataclasses import dataclass


def fill_first_n_dims(val: float, arr: np.ndarray, n: int)-> np.ndarray:
    '''
    create an nd-array filled with the provided value of the same shape 
    of the first n dimensions of the provided array arr
    '''
    if not isinstance(arr, np.ndarray):
        raise TypeError('input arr must be a NumPy ndarray')
    if not isinstance(n, int) or n < 0 or n > arr.ndim:
        raise ValueError(f'n must be between 0 and {arr.ndim}')

    new_shape = arr.shape[:n]
    return np.ones(new_shape) * val


def collapse_last_dim(val: float, arr: np.ndarray) -> np.ndarray:
    new_shape = list(arr.shape)
    new_shape[-1] = 1
    return np.ones(new_shape) * val


@dataclass(frozen=True)
class SimulatedPrices:
    log_returns: np.ndarray
    prices: np.ndarray


def simulate_prices(init_price: float, log_returns: np.ndarray):
    '''
    Let the log returns be r1, r2, r3, ...
    Let the initial price be S0, 
    The end result of generated prices should be
    S0 = S0 * exp(0)
    S1 = S0 * exp(0) * exp(r1) = S0 * exp(r1) 
    S2 = S0 * exp(0) * exp(r1) * exp(r2) = S0 * exp(r1 + r2)
    ...
    '''
    zeros = collapse_last_dim(0.0, log_returns)
    collapsed_index = log_returns.ndim - 1
    prices = init_price * np.exp(
        np.concatenate(
            [zeros, np.cumsum(log_returns, axis=collapsed_index)], 
            axis=collapsed_index)
    )
    return SimulatedPrices(log_returns=log_returns, prices=prices)


def simulate_gbm(
        gbm_params: gbm.GBMParameters,
        init_price: float, 
        shape: Tuple[int, ...], 
        dt: float, 
        rng: np.random.Generator,
):
    return simulate_prices(init_price, gbm.log_returns(gbm_params, shape, dt, rng))