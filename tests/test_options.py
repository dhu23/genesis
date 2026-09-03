from genesis import options
import numpy as np

def test_evaluate_european_payoff_by_monte_carlo():
    prices = np.array(
        [
            [100.        ,  27.17597334,  21.01387969,   2.43230102],
            [100.        , 116.89487932, 222.45572349,  85.8084154 ],
            [100.        ,  93.28468361,  72.71796756,  29.73237279],
            [100.        ,  25.60442839,   2.03442433,   5.05630701],
        ]
    )
    maturies = np.array([2, 3])
    strikes = np.array([95, 100, 105])

    ret = options.evaluate_european_payoff_by_price_paths(
        options.OptionsSide.Buy,
        options.OptionsType.Call,
        prices,
        maturies,
        strikes,
    )

    # print('payoff output', ret, ret.shape)
    expected_output = np.array(
        [
            [31.86393087, 30.61393087, 29.36393087],
            [ 0.,          0.,          0.,        ],
        ]
    )

    np.testing.assert_allclose(ret, expected_output)