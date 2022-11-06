import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def get_measures(
    var_prior,
    var_likelihood,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lambs,
    n_prior_samples=5000,
    n_post_samples=5000,
):
    results = {}
    results["bayes_risk"] = []  # expected log risk of bayesian posterior predictive
    results["gibbs_risk"] = []  # expected gibbs risk
    results["emp_gibbs_risk"] = []  # empirical gibbs risk
    results["kl"] = []  # kl between posterior and prior
    results["var_prior_pred"] = []  # prior predictive variance
    results["lamb_optimal"] = []  # optimal lamb according to eq.10 in draft.tex
    results["grad_expected_gibbs"] = []  # covariance in slides.tex
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

        # compute log_p
        samples_post = p_post.sample((n_post_samples,))
        log_p_test = Normal(
            X_test @ samples_post.T, torch.sqrt(var_likelihood)
        ).log_prob(Y_test)
        log_p_train = Normal(
            X_train @ samples_post.T, torch.sqrt(var_likelihood)
        ).log_prob(Y_train)

        # get bayesian generalization loss
        bayes_risk = (
            (torch.log(torch.tensor(n_post_samples)) - torch.logsumexp(log_p_test, 1))
            .mean()
            .item()
        )
        results["bayes_risk"].append(bayes_risk)

        # get expected gibbs loss
        gibbs_risk = -log_p_test.mean().item()
        results["gibbs_risk"].append(gibbs_risk)

        # get empirical gibbs loss
        emp_gibbs_risk = -log_p_train.mean().item()
        results["emp_gibbs_risk"].append(emp_gibbs_risk)

        # prior predictive variance
        p_prior = MultivariateNormal(torch.zeros(d_x), torch.eye(d_x) * var_prior)
        samples_prior = p_prior.sample((n_prior_samples,))
        n_D_n = n_test / n_train  # number of D_n sampled
        samples_X_n = torch.tensor(
            np.asarray(np.vsplit(X_test.numpy(), n_D_n))
        )  # n_D_n x n_train x d_x
        samples_Y_n = torch.tensor(
            np.asarray(np.vsplit(Y_test.numpy(), n_D_n))
        )  # n_D_n x n_train x 1
        log_p_prior = Normal(
            samples_X_n @ samples_prior.T, torch.sqrt(var_likelihood)
        ).log_prob(
            samples_Y_n
        )  # n_D_n x n_train x n_prior_samples
        var_prior_pred = torch.mean(
            torch.var(
                torch.sum(
                    log_p_prior,
                    1,
                ),
                0,
            )
        ).item()
        results["var_prior_pred"].append(var_prior_pred)

        # kl between p^lamb and prior
        kl = torch.distributions.kl.kl_divergence(p_post, p_prior).item()
        results["kl"].append(kl)

        # compute lamb^*
        lamb_optimal = torch.sqrt(
            2 * (kl + torch.log(torch.tensor(20.0))) / var_prior_pred
        ).item()
        results["lamb_optimal"].append(lamb_optimal)

        # derivative of expected gibbs loss
        grad_expected_gibbs = (
            -n_train
            * (
                (log_p_train.mean(0) * log_p_test.mean(0)).mean()
                - log_p_test.mean() * log_p_train.mean()
            ).item()
        )
        results["grad_expected_gibbs"].append(grad_expected_gibbs)
    return results
