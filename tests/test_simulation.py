from genesis import (
    simulation,
    gbm,
)
import numpy as np
import pytest

def test_fill_first_n_dims():
    sample = np.random.randint(0, 10, (1, 2, 3, 4))

    ret_1dim = simulation.fill_first_n_dims(1, sample, 1)
    np.testing.assert_allclose(ret_1dim, 1)
    assert ret_1dim.shape == (1,)

    ret_2dim = simulation.fill_first_n_dims(0, sample, 2)
    np.testing.assert_allclose(ret_2dim, [[0, 0]])
    assert ret_2dim.shape == (1, 2)

    ret_3dim = simulation.fill_first_n_dims(1, sample, 3)
    np.testing.assert_allclose(ret_3dim, [[[1, 1, 1], [1, 1, 1]]])
    assert ret_3dim.shape == (1, 2, 3)

    ret_4dim = simulation.fill_first_n_dims(2, sample, 4)
    expected_4dim = [[
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
    ]]
    np.testing.assert_allclose(ret_4dim, expected_4dim)
    assert ret_4dim.shape == (1, 2, 3, 4)

    # generates exceptions for large dimensions
    for n_dim in [5, 50, 500]:
        with pytest.raises(ValueError):
            simulation.fill_first_n_dims(2, sample, n_dim)


@pytest.mark.parametrize(
    'sample, expected',
    [
        (
            np.array([1, 2, 3, 4]), 
            np.array([0]),
        ),
        (
            np.array([[1, 2], [2, 3], [3, 4]]),
            np.array([[0], [0], [0]]),
        ),
        (
            np.array(
                [
                    [[1, 2, 3], [2, 3, 4]], 
                    [[2, 3, 4], [3, 4, 5]], 
                    [[3, 4, 5], [4, 5, 6]],
                ]
            ),
            np.array(
                [
                    [[0], [0]],
                    [[0], [0]],
                    [[0], [0]],
                ]
            ),
        )
    ]
)
def test_collapse_last_dimension(sample: np.ndarray, expected: np.ndarray):
    collapsed = simulation.collapse_last_dim(0, sample) 
    # print('testing collapse')
    # print(expected)
    # print(collapsed)   
    np.testing.assert_allclose(expected, collapsed)


@pytest.mark.parametrize(
        'sample_log_returns, expected_log_returns',
        [
            (
                np.array([0.01, 0.02, 0.03, 0.04]),
                np.array([0, 0.01, 0.03, 0.06, 0.10]),
            ),
            (
                np.array(
                    [
                        [0.01, 0.02, 0.03, 0.04], 
                        [0.01, 0.03, 0.05, 0.07], 
                        [-0.01, 0.01, 0.02, -0.03],
                    ]
                ),
                np.array(
                    [
                        [0.0, 0.01, 0.03, 0.06, 0.10],
                        [0.0, 0.01, 0.04, 0.09, 0.16],
                        [0.0, -0.01, 0.0, 0.02, -0.01],
                    ]
                )
            ),
            (
                np.array(
                    [
                        [
                            [0.01, 0.02, 0.03, 0.04],
                            [-0.01, -0.02, -0.03, -0.04],
                        ],
                        [
                            [-0.01, 0.02, -0.03, 0.04],
                            [0.01, -0.02, 0.03, -0.04],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.0, 0.01, 0.03, 0.06, 0.10],
                            [0.0, -0.01, -0.03, -0.06, -0.10],
                        ],
                        [
                            [0.0, -0.01, 0.01, -0.02, 0.02],
                            [0.0, 0.01, -0.01, 0.02, -0.02],
                        ],
                    ]
                ),
            )
        ]
)
def test_simulate_prices(
        sample_log_returns: np.ndarray,
        expected_log_returns: np.ndarray
):
    init_price: float = 2
    ret = simulation.simulate_prices(init_price, sample_log_returns)
    expected = init_price * np.exp(expected_log_returns)
    np.testing.assert_allclose(sample_log_returns, ret.log_returns)
    np.testing.assert_allclose(expected, ret.prices)


def test_simulate_gbm():
    params = gbm.GBMParameters(0.0, 1.0)
    init_px = 100.0
    rng = np.random.default_rng(9)
    ret = simulation.simulate_gbm(params, init_px, (4, 3), 1.0, rng)

    expected = simulation.SimulatedPrices(
        log_returns=np.array(
            [
                [-1.30283694, -0.25715009, -2.15634543],
                [ 0.15610488,  0.64345302, -0.952611  ],
                [-0.06951425, -0.24906743, -0.89435206],
                [-1.36240487, -2.53255243,  0.91042348],
            ]
        ),
        prices=np.array(
            [
                [100.        ,  27.17597334,  21.01387969,   2.43230102],
                [100.        , 116.89487932, 222.45572349,  85.8084154 ],
                [100.        ,  93.28468361,  72.71796756,  29.73237279],
                [100.        ,  25.60442839,   2.03442433,   5.05630701]
            ]
        ),
    )

    np.testing.assert_allclose(ret.log_returns, expected.log_returns)
    np.testing.assert_allclose(ret.prices, expected.prices)
    