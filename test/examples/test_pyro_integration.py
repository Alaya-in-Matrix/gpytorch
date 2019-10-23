#!/usr/bin/env python3

from math import pi
import os
import random
import torch
import pyro
import unittest
import gpytorch
from pyro.poutine.runtime import _PYRO_STACK


# Simple training data: let's try to learn sine and cosine functions
train_x = torch.linspace(0, 1, 100)

# y1 and y4 functions are sin(2*pi*x) with noise N(0, 0.05)
train_y1 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.05
train_y4 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.05
# y2 and y3 functions are -sin(2*pi*x) with noise N(0, 0.05)
train_y2 = -torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.05
train_y3 = -torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.05
# Create a train_y which interleaves the four
train_y = torch.stack([train_y1, train_y2, train_y3, train_y4], -1)


class ClusterGaussianLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, num_tasks, num_clusters, name_prefix=""):
        super().__init__()
        self.register_buffer("prior_cluster_logits", torch.zeros(num_tasks, num_clusters))
        self.register_buffer("temperature", torch.tensor(0.2))
        self.register_parameter("variational_cluster_logits", torch.nn.Parameter(torch.zeros(num_tasks, num_clusters)))
        self.register_parameter("raw_noise", torch.nn.Parameter(torch.tensor(0.)))
        self.num_tasks = num_tasks
        self.num_clusters = num_clusters
        self.name_prefix = name_prefix
        self.max_plate_nesting = 2

    @property
    def noise(self):
        return torch.nn.functional.softplus(self.raw_noise)

    def _cluster_dist(self, logits):
        dist = pyro.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)
        return dist

    def guide(self, **kwargs):
        with pyro.plate(self.name_prefix + ".trajectories_plate", self.variational_cluster_logits.size(-2), dim=-2):
            foo = pyro.sample(
                self.name_prefix + ".cluster_logits",
                self._cluster_dist(self.variational_cluster_logits.unsqueeze(-2))
            )

    def forward(self, function_samples, *params, **kwargs):
        with pyro.plate(self.name_prefix + ".trajectories_plate", self.prior_cluster_logits.size(-2), dim=-2):
            cluster_assignment_samples = pyro.sample(
                self.name_prefix + ".cluster_logits",
                self._cluster_dist(self.prior_cluster_logits.unsqueeze(-2))
            )
            print(self.variational_cluster_logits)
            print(cluster_assignment_samples.mean(0))
            print((cluster_assignment_samples.shape, function_samples.shape))
            print((cluster_assignment_samples.squeeze(-2) @ function_samples).transpose(-1, -2).shape)
            return pyro.distributions.Normal(
                loc=(cluster_assignment_samples.squeeze(-2) @ function_samples).transpose(-1, -2),
                scale=self.noise.sqrt(),
            ).to_event(1)


class ClusterMultitaskGPModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, train_x, train_y, num_functions=2):
        num_data = train_y.size(-2)
        
        # Define all the variational stuff
        inducing_points = torch.linspace(0, 1, 32).unsqueeze(-1).repeat(num_functions, 1, 1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=torch.Size([num_functions])
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Define likelihood
        self.likelihood = ClusterGaussianLikelihood(train_y.size(-1), num_functions, name_prefix="likelihood")

        # Mean, covar
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_functions])),
            batch_shape=torch.Size([num_functions]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return res


class TestPyroIntegration(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_multitask_gp_mean_abs_error(self):
        model = ClusterMultitaskGPModel(train_x, train_y)
        # Find optimal model hyperparameters
        model.train()

        # The MLL defines what type of approximate inference we're doing
        mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_x.size(0), beta=0.01)

        # Use the adam optimizer
        optimizer = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO(num_particles=256, vectorize_particles=True)
        svi = pyro.infer.SVI(mll.model, mll.guide, optimizer, elbo)

        n_iter = 100
        for _ in range(n_iter):
            svi.step(train_x, train_y)

        # Test the model
        with gpytorch.settings.num_likelihood_samples(128):
            model.eval()
            test_x = torch.linspace(0, 1, 51)
            test_y1 = torch.sin(test_x * (2 * pi))
            test_y2 = -torch.sin(test_x * (2 * pi))
            test_y3 = -torch.sin(test_x * (2 * pi))
            test_y4 = torch.sin(test_x * (2 * pi))
            test_preds = model.likelihood(model(test_x)).mean
            mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds.mean(0)[:, 0]))
            mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds.mean(0)[:, 1]))
            mean_abs_error_task_3 = torch.mean(torch.abs(test_y3 - test_preds.mean(0)[:, 2]))
            mean_abs_error_task_4 = torch.mean(torch.abs(test_y4 - test_preds.mean(0)[:, 3]))
            print(test_preds[:, 3].shape, test_y4.shape)
            print(mean_abs_error_task_2)

        self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_3.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_4.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
