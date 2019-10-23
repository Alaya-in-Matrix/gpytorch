#!/usr/bin/env python3

import warnings
import pyro
from .abstract_variational_gp import AbstractVariationalGP


class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix="", beta=1.0):
        # Deprecated for 0.4 release
        warnings.warn(
            "PyroVariationalGP is deprecated. Please use a VariationalGP/AbstractGP "
            "and use the model and guides that are part of a "
            "gpytorch.mlls._ApproximateLogLikelihood.", DeprecationWarning
        )
        super(PyroVariationalGP, self).__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data
        self.beta = beta

    def guide(self, input, output, *args, **kwargs):
        # Draw samples from q(u) for KL divergence computation
        with pyro.poutine.scale(scale=float(self.beta)):
            u_distribution = self.variational_strategy.variational_distribution
            pyro.sample(self.name_prefix + ".u", u_distribution)
        # Draw samples from the likelihood's guide
        pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape
        self.likelihood.guide(*args, **kwargs)

    def model(self, input, target, *args, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)

        # Draw samples from p(u) for KL divergence computation
        with pyro.poutine.scale(scale=float(self.beta)):
            u_distribution = self.variational_strategy.prior_distribution
            pyro.sample(self.name_prefix + ".u", u_distribution)

        # Get the variational distribution for the function
        f_distribution = self(input)

        # Draw samples from the likelihood
        num_minibatch = f_distribution.batch_shape[-1]
        with pyro.poutine.scale(scale=float(self.num_data / num_minibatch)):
            f_distribution = pyro.distributions.Normal(loc=f_distribution.loc, scale=f_distribution.variance.sqrt())
            return self.likelihood.pyro_sample_output(target, f_distribution, *args, **kwargs)
