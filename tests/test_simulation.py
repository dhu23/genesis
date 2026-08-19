from genesis import simulation
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


def test_simulate_prices():
    pass