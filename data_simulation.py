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


def get_data_specified(
    d_x=101,
    n_data=100,
):
    """model is well-specified.
    data generating process is v(y|x)=p(y|x,theta)=N(fourier(x)^T@theta,variance),
    where theta=1
    """
    variance = 1.0 / 40.0
    d_y = 1

    X_orig = np.random.uniform(-1.0, 1.0, n_data)
    X = np.asarray(get_fourier_basis(X_orig, d_x))
    Y = np.random.normal(
        np.sum(X, 1).reshape(n_data, 1), np.sqrt(variance), (n_data, 1)
    )

    assert X.shape == (n_data, d_x)
    assert Y.shape == (n_data, d_y)
    assert X_orig.shape == (n_data,)

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_orig).float(),
    )


def get_data_misspecified(
    d_x=101,
    n_data=100,
):
    """model is misspecified because data is heterogeneous while
    the model class is homogeneous.
    """
    y = 1.0  # mean of the data generating process
    d_y = 1
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

    assert X.shape == (n_data, d_x)
    assert Y.shape == (n_data, d_y)
    assert X_orig.shape == (n_data,)

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_orig).float(),
    )


# def get_X_show_grunwald(d_x=101, n_data=10000):
#     """handy for making model fit plot"""
#     X_orig = np.linspace(-1.0, 1.0, n_data)
#     X = np.asarray(get_fourier_basis(X_orig, d_x))

#     assert X.shape == (n_data, d_x)
#     assert X_orig.shape == (n_data,)

#     return torch.from_numpy(X).float(), torch.from_numpy(X_orig).float()
