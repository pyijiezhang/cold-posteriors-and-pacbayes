import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def get_metrics(
    var_prior,
    var_likelihood,
    X_train,
    Y_train,
    X_train_orig,
    X_test,
    Y_test,
    X_test_orig,
    lambs,
    gammas,
    n_post_samples=5000,
):
    results = {}

    results["nll_bayes_test"] = []
    results["nll_bayes_train"] = []
    results["nll_gibbs_test"] = []
    results["nll_gibbs_train"] = []

    results["mse_bayes_test"] = []
    results["mse_bayes_train"] = []
    results["mse_gibbs_test"] = []
    results["mse_gibbs_train"] = []

    results["grad_expected_gibbs"] = []

    if lambs == None:
        lamb = 1.0
        for gamma in gammas:

            print(gamma)

            d_x = X_train.shape[1]
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            # settings
            # prior
            var_prior = torch.as_tensor(1 / gamma)
            # likelihood
            var_likelihood = torch.as_tensor(var_likelihood)

            # compute posterior distribution, see bishop eq 3.53 and 3.54
            # X_train is the design matrix, n_train by d_x matrix
            alpha = var_prior ** (-1)  # prior recision
            beta_orig = var_likelihood ** (
                -1
            )  # gaussian likelihood precision before absorbing temperature
            beta = (
                beta_orig * lamb
            )  # gaussian likelihood precision after absorbing temperature
            S_N_inv = (
                alpha * torch.eye(d_x) + beta * X_train.T @ X_train
            )  # posterior precision
            S_N = torch.inverse(S_N_inv)  # posterior variance
            m_N = (beta * S_N @ X_train.T @ Y_train).reshape(-1)  # posterior mean
            p_post = MultivariateNormal(m_N, precision_matrix=S_N_inv)

            # log_p:n_test,n_post_samples
            # samples_post:n_post_samples,d_x
            samples_post = p_post.sample((n_post_samples,))
            log_p_test = Normal(
                X_test @ samples_post.T, torch.sqrt(var_likelihood)
            ).log_prob(Y_test)
            log_p_train = Normal(
                X_train @ samples_post.T, torch.sqrt(var_likelihood)
            ).log_prob(Y_train)

            nll_bayes_test = (
                (
                    torch.log(torch.tensor(n_post_samples))
                    - torch.logsumexp(log_p_test, 1)
                )
                .mean()
                .item()
            )
            results["nll_bayes_test"].append(nll_bayes_test)

            nll_bayes_train = (
                (
                    torch.log(torch.tensor(n_post_samples))
                    - torch.logsumexp(log_p_train, 1)
                )
                .mean()
                .item()
            )
            results["nll_bayes_train"].append(nll_bayes_train)

            nll_gibbs_test = -log_p_test.mean().item()
            results["nll_gibbs_test"].append(nll_gibbs_test)

            nll_gibbs_train = -log_p_train.mean().item()
            results["nll_gibbs_train"].append(nll_gibbs_train)

            mse_bayes_test = (
                (((X_test @ samples_post.mean(0)).mean() - X_test_orig) ** 2)
                .mean()
                .item()
            )
            mse_bayes_train = (
                (((X_train @ samples_post.mean(0)).mean() - X_train_orig) ** 2)
                .mean()
                .item()
            )
            mse_gibbs_test = (
                (((samples_post @ X_test.T).mean(0) - X_test_orig) ** 2).mean().item()
            )
            mse_gibbs_train = (
                (((samples_post @ X_train.T).mean(0) - X_train_orig) ** 2).mean().item()
            )

            results["mse_bayes_test"].append(mse_bayes_test)
            results["mse_bayes_train"].append(mse_bayes_train)
            results["mse_gibbs_test"].append(mse_gibbs_test)
            results["mse_gibbs_train"].append(mse_gibbs_train)

            grad_expected_gibbs = (
                -n_train
                * (
                    (log_p_train.mean(0) * log_p_test.mean(0)).mean()
                    - log_p_test.mean() * log_p_train.mean()
                ).item()
            )
            results["grad_expected_gibbs"].append(grad_expected_gibbs)
    if gammas == None:
        for lamb in lambs:

            print(lamb)

            d_x = X_train.shape[1]
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            # settings
            # prior
            var_prior = torch.as_tensor(var_prior)
            # likelihood
            var_likelihood = torch.as_tensor(var_likelihood)

            # compute posterior distribution, see bishop eq 3.53 and 3.54
            # X_train is the design matrix, n_train by d_x matrix
            alpha = var_prior ** (-1)  # prior recision
            beta_orig = var_likelihood ** (
                -1
            )  # gaussian likelihood precision before absorbing temperature
            beta = (
                beta_orig * lamb
            )  # gaussian likelihood precision after absorbing temperature
            S_N_inv = (
                alpha * torch.eye(d_x) + beta * X_train.T @ X_train
            )  # posterior precision
            S_N = torch.inverse(S_N_inv)  # posterior variance
            m_N = (beta * S_N @ X_train.T @ Y_train).reshape(-1)  # posterior mean
            p_post = MultivariateNormal(m_N, precision_matrix=S_N_inv)

            # log_p:n_test,n_post_samples
            # samples_post:n_post_samples,d_x
            samples_post = p_post.sample((n_post_samples,))
            log_p_test = Normal(
                X_test @ samples_post.T, torch.sqrt(var_likelihood)
            ).log_prob(Y_test)
            log_p_train = Normal(
                X_train @ samples_post.T, torch.sqrt(var_likelihood)
            ).log_prob(Y_train)

            nll_bayes_test = (
                (
                    torch.log(torch.tensor(n_post_samples))
                    - torch.logsumexp(log_p_test, 1)
                )
                .mean()
                .item()
            )
            results["nll_bayes_test"].append(nll_bayes_test)

            nll_bayes_train = (
                (
                    torch.log(torch.tensor(n_post_samples))
                    - torch.logsumexp(log_p_train, 1)
                )
                .mean()
                .item()
            )
            results["nll_bayes_train"].append(nll_bayes_train)

            nll_gibbs_test = -log_p_test.mean().item()
            results["nll_gibbs_test"].append(nll_gibbs_test)

            nll_gibbs_train = -log_p_train.mean().item()
            results["nll_gibbs_train"].append(nll_gibbs_train)

            mse_bayes_test = (
                (((X_test @ samples_post.mean(0)).mean() - X_test_orig) ** 2)
                .mean()
                .item()
            )
            mse_bayes_train = (
                (((X_train @ samples_post.mean(0)).mean() - X_train_orig) ** 2)
                .mean()
                .item()
            )
            mse_gibbs_test = (
                (((samples_post @ X_test.T).mean(0) - X_test_orig) ** 2).mean().item()
            )
            mse_gibbs_train = (
                (((samples_post @ X_train.T).mean(0) - X_train_orig) ** 2).mean().item()
            )

            results["mse_bayes_test"].append(mse_bayes_test)
            results["mse_bayes_train"].append(mse_bayes_train)
            results["mse_gibbs_test"].append(mse_gibbs_test)
            results["mse_gibbs_train"].append(mse_gibbs_train)

            grad_expected_gibbs = (
                -n_train
                * (
                    (log_p_train.mean(0) * log_p_test.mean(0)).mean()
                    - log_p_test.mean() * log_p_train.mean()
                ).item()
            )
            results["grad_expected_gibbs"].append(grad_expected_gibbs)

    return results
