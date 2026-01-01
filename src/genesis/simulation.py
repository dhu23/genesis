# Generate simulated prices

from typing import NamedTuple
import numpy as np
import genesis.gbm as gbm


class SimulatedPrices(NamedTuple):
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
    prices = init_price * np.exp(np.concatenate([[0.], np.cumsum(log_returns)]))
    return SimulatedPrices(log_returns=log_returns, prices=prices)


def simulate_gbm(
        gbm_params: gbm.GBMParameters,
        init_price: float, 
        n: int, 
        dt: float, 
        rng: np.random.Generator,
):
    return simulate_prices(init_price, gbm.log_returns(gbm_params, n, dt, rng))