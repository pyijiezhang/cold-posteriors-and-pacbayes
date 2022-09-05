import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def get_measures(var_prior, var_likelihood, X_train, Y_train, X_test, Y_test, lambs):
    log_risks = []  # expected risk bayesian posterior predictive
    r_log_risks = []  # expected gibbs risk
    emp_r_log_risks = []  # empirical gibbs risk
    kls = []  # kl between posterior and prior
    vars_prior_pred = []  # prior predictive variance
    for lamb in lambs:

        print(lamb)

        d_x = X_train.shape[1]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # settings
        # prior
        var_prior = torch.tensor(var_prior)
        n_prior_samples = 5000  # no. of samples from prior
        # likelihood
        var_likelihood = torch.tensor(var_likelihood)
        # posterior
        n_post_samples = 5000  # no. of samples from posterior

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

        # computre posterior predictive distribution, see bishop eq 3.58 and 3.59
        mean_post_pred = X_test @ m_N
        cov_post_pred = torch.eye(n_test) / beta + X_test @ S_N @ X_test.T
        std_post_pred = torch.sqrt(torch.diag(cov_post_pred))
        p_post_pred = Normal(mean_post_pred, std_post_pred)

        # get bayesian generalization loss
        log_risk = -p_post_pred.log_prob(Y_test).mean().item()
        # log_risk1 = (
        #     -torch.log(
        #         torch.exp(
        #             Normal(
        #                 X_test @ samples_post.T, torch.sqrt(var_likelihood / lamb)
        #             ).log_prob(Y_test)
        #         ).mean(1)
        #     )
        #     .mean()
        #     .item()
        # )
        log_risks.append(log_risk)

        # get expected gibbs loss
        samples_post = p_post.sample((n_post_samples,))
        r_log_risk = (
            -Normal(X_test @ samples_post.T, torch.sqrt(var_likelihood / lamb))
            .log_prob(Y_test)
            .mean()
            .item()
        )
        r_log_risks.append(r_log_risk)

        # get empirical gibbs loss
        emp_r_log_risk = (
            -Normal(X_train @ samples_post.T, torch.sqrt(var_likelihood / lamb))
            .log_prob(Y_train)
            .mean()
            .item()
        )
        emp_r_log_risks.append(emp_r_log_risk)

        # prior predictive variance
        p_prior = MultivariateNormal(torch.zeros(d_x), torch.eye(d_x) * var_prior)
        samples_prior = p_prior.sample((n_prior_samples,))
        samples_X_n = torch.tensor(
            np.asarray(np.vsplit(X_test.numpy(), n_test / n_train))
        )
        samples_Y_n = torch.tensor(
            np.asarray(np.vsplit(Y_test.numpy(), n_test / n_train))
        )
        var_prior_pred = torch.mean(
            torch.var(
                torch.sum(
                    Normal(
                        samples_X_n @ samples_prior.T, torch.sqrt(var_likelihood)
                    ).log_prob(samples_Y_n),
                    1,
                ),
                1,
            )
        ).item()
        vars_prior_pred.append(var_prior_pred)

        # kl
        kl = torch.distributions.kl.kl_divergence(p_post, p_prior).item()
        kls.append(kl)

    return log_risks, r_log_risks, emp_r_log_risks, kls, vars_prior_pred
