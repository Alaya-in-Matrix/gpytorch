#!/usr/bin/env python3

import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from abc import ABC, abstractmethod
from .. import settings


try:
    import pyro
    pyro.factor

    class _PyroMLLMixin(object):
        def guide(self, input, target, **kwargs):
            # Hack to get the current plating structure
            pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape
            return self.likelihood.guide(**kwargs)

        def model(self, input, target, **kwargs):
            pyro.module(self.name_prefix + ".gp_prior", self.gp_model)
            pyro.module(self.name_prefix + ".likelihood", self.likelihood)

            # Compute q(f)
            approximate_dist_f = self.gp_model(input)

            # Get the components of the log likelihood
            combine_terms = self.combine_terms  # We're going to temporarily change this term
            self.combine_terms = False
            components = self.forward(approximate_dist_f, target, **kwargs)
            self.combine_terms = combine_terms

            if len(components) == 3:
                log_likelihood, kl_divergence, log_prior = components
            else:
                log_likelihood, kl_divergence, log_prior, added_loss_term = components
                pyro.factor(self.name_prefix + ".added_loss_term", added_loss_term)

            pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)
            pyro.factor(self.name_prefix + ".kl_divergence", kl_divergence)
            pyro.factor(self.name_prefix + ".log_prior", log_prior)

except ImportError:
    class _PyroMLLMixin(object):
        def guide(self, approximate_dist_f, target, **kwargs):
            raise RuntimeError("Cannot use model/guide methods because you dont have Pyro installed.")

        def model(self, approximate_dist_f, target, **kwargs):
            raise RuntimeError("Cannot use model/guide methods because you dont have Pyro installed.")

except AttributeError:  # We need pyro.factor
    class _PyroMLLMixin(object):
        def guide(self, approximate_dist_f, target, **kwargs):
            raise RuntimeError(f"Pyro compatibility requires Pyro version >=0.4.2. You have {pyro.__version__}.")

        def model(self, approximate_dist_f, target, **kwargs):
            raise RuntimeError(f"Pyro compatibility requires Pyro version >=0.4.2. You have {pyro.__version__}.")


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, _PyroMLLMixin, ABC):
    r"""
    An approximate marginal log likelihood (typically a bound) for approximate GP models.
    We expect that :attr:`model` is a :obj:`gpytorch.models.ApproximateGP`.

    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
        :attr:`combine_terms` (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)
        :attr:`name_prefix` (str):
            (For Pyro integration) - supplied name to be part of the prefix.
    """
    def __init__(self, likelihood, model, num_data, beta=1., combine_terms=True, name_prefix=""):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta
        self.name_prefix = name_prefix

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        raise NotImplementedError

    def _kl_divergence(self):
        with settings.max_preconditioner_size(0):
            prior_dist = self.gp_model.variational_strategy.prior_distribution
            variational_dist_u = self.gp_model.variational_strategy.variational_distribution
            kl_divergence = torch.distributions.kl.kl_divergence(variational_dist_u, prior_dist)
        return kl_divergence

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape.numel()
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs).div(num_batch)
        kl_divergence = self._kl_divergence().div(self.num_data / self.beta)

        # Make sure LL and KL terms are the same size
        if log_likelihood.numel() == 1:
            kl_divergence = kl_divergence.sum()
        elif kl_divergence.dim() > log_likelihood.dim():
            kl_divergence = kl_divergence.sum(-1)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.gp_model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for _, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure()).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data), added_loss
            else:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data)

    def guide(self, input, target, **kwargs):
        r"""
        """
        super().guide(input, target, **kwargs)

    def model(self, input, target, **kwargs):
        r"""
        """
        super().model(input, target, **kwargs)
