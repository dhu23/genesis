# all options pricing related
from enum import Enum
import numpy as np


class OptionsSide(Enum):
    Buy = 0
    Sell = 1

class OptionsType(Enum):
    Call = 0
    Put = 1

# Monte Carlo Based
# calculate European option payoff by averaging payoff over all paths

def evaluate_european_payoff_by_price_paths(
        side: OptionsSide, 
        option_type: OptionsType,
        price_path: np.ndarray, # a 2-D array, the last index is price point
        maturies: np.ndarray, # a one dimensional array for maturity index
        strikes: np.ndarray, # a one dimensional array for strike values
):
    if np.ndim(maturies) != 1:
        raise ValueError(f'maturies ndim should be 1')
    if np.ndim(strikes) != 1:
        raise ValueError(f'strikes ndim should be 1')

    if np.ndim(price_path) != 2:
        raise ValueError(f'price path ndim should be 2')

    # retrieve price offsets relevant to 

    # add a new index to the offset price end, so its (px, py, 1) 
    # add a new index to strikes, so its (1, s)

    x = price_path[:, maturies]
    # print('prices on maturies', x, x.shape)

    # print('strikes', strikes, strikes.shape)
    
    basic_payoffs = price_path[:, maturies][:, :, None] - strikes
    if option_type == OptionsType.Call:
        payoffs = np.maximum(basic_payoffs, 0)
    else:
        payoffs = np.maximum(-basic_payoffs, 0)
    sign = 1 if side == OptionsSide.Buy else -1
    # print('payoffs', payoffs, payoffs.shape)
    return payoffs.mean(axis=0) * sign