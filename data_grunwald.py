import numpy as np
import torch


def fourier_basis(x, p):
    res = []
    for i in range(0, p):
        k = (i + 1) // 2
        if i == 0:
            res.append(1 / np.sqrt(2 * np.pi))
        elif (i + 1) % 2 == 0:
            res.append(1 / np.sqrt(np.pi) * np.cos(k * x))
        else:
            res.append(1 / np.sqrt(np.pi) * np.sin(k * x))
    return np.array(res)


def get_fourier_basis(x, p):
    res = np.zeros((x.shape[0], p))
    for i in range(x.shape[0]):
        res[i, :] = fourier_basis(x[i], p)
    return res


def get_data(
    y=0.0,
    d_x=101,
    n_data=100,
    if_misspecified=True,
):
    d_y = 1

    if if_misspecified:
        variance = 2 * 1.0 / 40.0
        n_easy = np.random.binomial(n_data, 0.5)
        X_orig = np.concatenate(
            [np.random.uniform(-1.0, 1.0, n_data - n_easy), np.zeros(n_easy)]
        )
        X = np.asarray(get_fourier_basis(X_orig, d_x))
        Y = np.concatenate(
            [
                np.random.normal(y, np.sqrt(variance), (n_data - n_easy, 1)),
                np.ones((n_easy, 1)) * y,
            ]
        )
    else:
        variance = 1.0 / 40.0
        X_orig = np.random.uniform(-1.0, 1.0, n_data)
        X = np.asarray(get_fourier_basis(X_orig, d_x))
        Y = np.random.normal(y, np.sqrt(variance), (n_data, 1))

    assert X.shape == (n_data, d_x)
    assert Y.shape == (n_data, d_y)
    assert X_orig.shape == (n_data,)

    return X, Y, X_orig


def get_X_show(d_x=101, n_data=10000):
    """handy for making model fit plot"""
    X_orig = np.linspace(-1.0, 1.0, n_data)
    X = np.asarray(get_fourier_basis(X_orig, d_x))

    assert X.shape == (n_data, d_x)
    assert X_orig.shape == (n_data,)

    return X, X_orig
