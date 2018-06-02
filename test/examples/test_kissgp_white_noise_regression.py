from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
import unittest
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel, WhiteNoiseKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable


# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
def make_data(cuda=False):
    train_x = Variable(torch.linspace(0, 1, 100))
    train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))
    test_x = Variable(torch.linspace(0, 1, 51))
    test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.grid_covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=50, grid_bounds=[(0, 1)])
        self.noise_covar_module = WhiteNoiseKernel(variances=torch.ones(100) * 0.001)
        self.covar_module = self.grid_covar_module + self.noise_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class TestKISSGPRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_kissgp_gp_mean_abs_error(self):
        train_x, train_y, test_x, test_y = make_data()
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x.data, train_y.data, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(25):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for param in likelihood.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        test_preds = likelihood(gp_model(test_x)).mean()
        mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.data.squeeze().item(), 0.05)

    def test_kissgp_gp_fast_pred_var(self):
        with gpytorch.fast_pred_var():
            train_x, train_y, test_x, test_y = make_data()
            likelihood = GaussianLikelihood()
            gp_model = GPRegressionModel(train_x.data, train_y.data, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(25):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            gp_model.eval()
            likelihood.eval()
            # Set the cache
            test_function_predictions = likelihood(gp_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.log_noise.data.fill_(3)
            test_function_predictions = likelihood(gp_model(train_x))

            noise = likelihood.log_noise.exp()
            var_diff = (test_function_predictions.var() - noise).abs()
            self.assertLess(torch.max(var_diff.data / noise.data), 0.05)

    def test_kissgp_gp_mean_abs_error_cuda(self):
        if torch.cuda.is_available():
            train_x, train_y, test_x, test_y = make_data(cuda=True)
            likelihood = GaussianLikelihood().cuda()
            gp_model = GPRegressionModel(train_x.data, train_y.data, likelihood).cuda()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(25):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            gp_model.eval()
            likelihood.eval()
            test_preds = likelihood(gp_model(test_x)).mean()
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

            self.assertLess(mean_abs_error.data.squeeze().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
