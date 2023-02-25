import logging
import time
import os

import gpytorch
import numpy as np
import pandas as pd
import torch
import tqdm
from gpytorch.kernels import InducingPointKernel
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from utils.utils import reshape_dim1, reshape_pt1

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Classes of GPyTorch models we use in this repo

# Tutorial: why and how to use pytorch
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://pytorch.org/docs/stable/optim.html

# Creating and copying tensors
# https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor

# WARNINGS:
# likelihood.noise should not be all too small, hurts performance (min 1e-4)

# GP models for use with GPyTorch
# https://docs.gpytorch.ai/en/v1.2.0/examples/01_Exact_GPs/Simple_GP_Regression.html

# Scalability for large datasets: several approaches are possible
# https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/index.html
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/index.html
# https://github.com/cornellius-gp/gpytorch/issues/1290
# SVGP models are slower to train but faster to predict and take less memory
# than SparseGP models

# CUDA memory issues
# https://discuss.pytorch.org/t/how-does-reserved-in-total-by-pytorch-work/70172/9

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        train_x = torch.squeeze(train_x)
        train_y = torch.squeeze(train_y)
        train_y = train_y.contiguous()
        # pytorch necessitates train targets to be contiguous:
        # https://stackoverflow.com/questions/48915810/pytorch-contiguous
        # https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays/26999092#26999092
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        # https://docs.gpytorch.ai/en/v1.1.1/kernels.html#composition-decoration-kernels

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, likelihood, train_x, train_y, training_iter=50,
                 optimizer=torch.optim.Adam, lr=0.1, optim_options={},
                 scheduler=None, scheduler_options={}, stopper=None,
                 parameters=None, verbose=True):
        train_x = torch.squeeze(train_x).double()
        train_y = torch.squeeze(train_y).double().contiguous()
        # https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html
        losses = []
        self.train()
        likelihood.train()
        if 'LBFGS' in optimizer.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('LBFGS is slow and memory intensive but '
                            'also converges well (with strong_wolfe). It does '
                            'not support gain scheduling and early stopping!')
            if stopper or scheduler:
                raise NotImplementedError
        if not parameters:
            parameters = self.parameters()
        if 'mll' in optim_options.keys():
            mll = optim_options['mll']
            optim_options = optim_options.copy()
            optim_options.pop('mll')
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood
        optimizer = optimizer(parameters, lr=lr, **optim_options)
        if scheduler:
            scheduler = scheduler(optimizer, verbose=False,
                                  **scheduler_options)
        mll = mll(likelihood, self)
        for i in range(training_iter):
            if 'LBFGS' in optimizer.__class__.__name__:
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    output = self(train_x)
                    loss = -mll(output, train_y)
                    losses.append(loss.item())
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                start = time.time()
                optimizer.step(closure)
                end = time.time()
                loss = losses[-1]
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss))
            else:
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(train_x)
                # Calculate loss and backpropagate gradients
                loss = -mll(output, train_y)
                losses.append(loss.item())
                loss.backward()
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()))
                optimizer.step()
                if scheduler:
                    scheduler.step(loss)
                if stopper:
                    stopper(loss)
                    if stopper.early_stop:
                        # Stop early
                        break
        return losses

    def predict(self, x, likelihood, full_cov=False, onlymean=False):
        self.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/728
            observed_pred = likelihood(self(x))
            if not full_cov:
                mean = observed_pred.mean
                if onlymean:
                    var = torch.zeros_like(mean, device=mean.device)
                else:
                    var = observed_pred.variance
            else:
                mean = observed_pred.mean
                if onlymean:
                    var = torch.zeros((mean.shape[1], mean.shape[1]),
                                      device=mean.device)
                else:
                    var = observed_pred.covariance_matrix
        # Reshape: if dx or dy = 1 then output will otherwise be vector!
        if len(x) == 1:
            mean = reshape_pt1(mean)
            var = reshape_pt1(var)
        elif len(x.shape) > 1:
            if len(mean.shape) == 1:
                mean = reshape_dim1(mean)
                var = reshape_dim1(var)
        return mean, var

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            mean = torch.autograd.grad(observed_pred.mean.sum(), test_x,
                                       retain_graph=True)[0]
            if onlymean:
                var = torch.zeros_like(mean)
            else:
                var = torch.autograd.grad(observed_pred.variance.sum(), test_x,
                                          retain_graph=True)[0]
        # Reshape: if dx = 1 then output will otherwise be vector!
        if len(x) == 1:
            mean = reshape_pt1(mean)
            var = reshape_pt1(var)
        elif len(x.shape) > 1:
            if len(mean.shape) == 1:
                mean = reshape_dim1(mean)
                var = reshape_dim1(var)
        return mean, var  # (N, dx)


# Sparse GPs
# https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html?highlight=sparse

class SparseGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel,
                 inducing_points=torch.tensor([])):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood,
                                            kernel)
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        self.base_covar_module = self.covar_module
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=inducing_points,
            likelihood=likelihood)

    def optimize(self, likelihood, train_x, train_y, training_iter=50,
                 optimizer=torch.optim.Adam, lr=0.1, optim_options={},
                 scheduler=None, scheduler_options={}, stopper=None,
                 parameters=None, verbose=False):
        if 'Adam' in optimizer.__class__.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('Adam does not seem great with sparse models?')
        super().optimize(
            likelihood, train_x, train_y, training_iter=training_iter,
            optimizer=optimizer, lr=lr, optim_options=optim_options,
            scheduler=scheduler, scheduler_options=scheduler_options,
            stopper=stopper, parameters=parameters, verbose=verbose)


# Multioutput GPs: Batch Independent Multitask GPs
# https://docs.gpytorch.ai/en/v1.1.1/examples/03_Multitask_Exact_GPs/index.html
# For this class only: differentiate between 2 likelihood noise values:
# https://github.com/cornellius-gp/gpytorch/issues/1303
# Set likelihood.noise_covar.noise to your true value, likelihood.noise to
# something small. For other classes set both to your true value.


class MultioutGPModel(ExactGPModel):
    # Multi output GPs with same type of kernel and likelihood for each
    # independent output dimension, but possibly different hyperparameters
    # https://docs.gpytorch.ai/en/v1.1.1/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html
    def __init__(self, train_x, train_y, likelihood, kernel, output_size=1):
        assert kernel.batch_shape[0] != 0, \
            'The kernel and likelihood have not been given the batch_shape ' \
            'parameter to indicate the number of tasks.'
        self.output_size = output_size
        super().__init__(train_x, train_y, likelihood, kernel)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([output_size]))
        self.output_size = output_size

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        multitask_result = \
            gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
        return multitask_result

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            observed_mean = observed_pred.mean
            observed_var = observed_pred.variance
            # Workaround in multi-output case, might be a bit slow...
            mean = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                           observed_mean.shape[1]]))
            var = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                          observed_var.shape[1]]))
            for i in range(observed_mean.shape[1]):
                mean[:, :, i] = torch.autograd.grad(
                    observed_mean[:, i].sum(), test_x, retain_graph=True)[0]
                if not onlymean:
                    var[:, :, i] = torch.autograd.grad(
                        observed_var[:, i].sum(), test_x, retain_graph=True)[0]
        return mean.transpose(1, 2), var.transpose(1, 2)  # (N, dy, dx)


# Multioutput sparse GPs (mixture of both)
# https://github.com/cornellius-gp/gpytorch/issues/1043
# https://docs.gpytorch.ai/en/v1.2.1/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html
# Principle: approximate K over set of fixed nb of inducing points, which are
# then optimized to maximize marginal log likelihood (or other)
# For this class only: sometimes for kernel hyperparameters, cannot both
# initialize kernel and set prior on lengthscale... Initializing kernel
# manually seems to work, i.e. setting value for each param, as for likelihood!

class MultioutSparseGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel,
                 inducing_points=torch.tensor([]), output_size=1):
        assert kernel.batch_shape[0] != 0, \
            'The kernel and likelihood have not been given the batch_shape ' \
            'parameter to indicate the number of tasks.'
        self.output_size = output_size
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        # Reshape for batch mode GP
        inducing_points = inducing_points.repeat(output_size, 1, 1)
        train_x = train_x.repeat(output_size, 1, 1)
        train_y = train_y.t()
        super().__init__(train_x, train_y, likelihood, kernel)
        self.base_covar_module = self.covar_module
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=inducing_points,
            likelihood=likelihood)

    def optimize(self, likelihood, train_x, train_y, training_iter=50,
                 optimizer=torch.optim.Adam, lr=0.1, optim_options={},
                 scheduler=None, scheduler_options={}, stopper=None,
                 parameters=None, verbose=False):
        train_x = torch.squeeze(train_x).double()
        train_y = torch.squeeze(train_y).double().contiguous()
        train_x = train_x.repeat(self.output_size, 1, 1)
        train_y = train_y.t()
        # https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html
        losses = []
        self.train()
        likelihood.train()
        if 'LBFGS' in optimizer.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('LBFGS is slow and memory intensive but '
                            'also converges well (with strong_wolfe). It does'
                            'not support gain scheduling and early stopping!')
            if stopper or scheduler:
                raise NotImplementedError
        if not parameters:
            parameters = self.parameters()
        if 'mll' in optim_options.keys():
            mll = optim_options['mll']
            optim_options = optim_options.copy()

        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood
        optimizer = optimizer(parameters, lr=lr, **optim_options)
        if scheduler:
            scheduler = scheduler(optimizer, verbose=False,
                                  **scheduler_options)
        elif 'Adam' in optimizer.__class__.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('Adam does not seem great with sparse models?')
        mll = mll(likelihood, self)
        for i in range(training_iter):
            if 'LBFGS' in optimizer.__class__.__name__:
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    output = self(train_x)
                    loss = -mll(output, train_y).sum()  # or mean()
                    losses.append(loss.item())
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                start = time.time()
                optimizer.step(closure)
                end = time.time()
                if verbose:
                    print('Iter %d/%d - Loss' % (i + 1, training_iter), 'in ',
                          str(end - start))
            else:
                start = time.time()
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(train_x)
                # Calculate loss and backpropagate gradients
                loss = -mll(output, train_y).sum()  # or mean()
                losses.append(loss.item())
                loss.backward()
                end = time.time()
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()), 'in ',
                          str(end - start))
                optimizer.step()
                if scheduler:
                    scheduler.step(loss)
                if stopper:
                    stopper(loss)
                    if stopper.early_stop:
                        # Stop early
                        break
        return losses

    def predict(self, x, likelihood, full_cov=False, onlymean=False):
        start = time.time()
        x = x.repeat(self.output_size, 1, 1)
        mid = time.time()
        mean, var = super().predict(x, likelihood, full_cov=full_cov,
                                    onlymean=onlymean)
        end = time.time()
        return mean.t(), var.t()

    def predict_gradient(self, x, likelihood, onlymean=False):
        x = x.repeat(self.output_size, 1, 1)
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            observed_mean = observed_pred.mean.t()
            observed_var = observed_pred.variance.t()
            # Workaround in multi-output case, might be a bit slow...
            mean = torch.zeros(torch.Size([len(observed_mean), test_x.shape[2],
                                           observed_mean.shape[1]]))
            var = torch.zeros(torch.Size([len(observed_var), test_x.shape[2],
                                          observed_var.shape[1]]))
            for i in range(observed_mean.shape[1]):
                mean[:, :, i] = torch.autograd.grad(
                    observed_mean[:, i].sum(), test_x, retain_graph=True)[0][i]
                if not onlymean:
                    var[:, :, i] = torch.autograd.grad(
                        observed_var[:, i].sum(), test_x, retain_graph=True)[0][i]
        return mean.transpose(1, 2), var.transpose(1, 2)  # (N, dy, dx)

# My_Multioutput sparse GPs (mixture of both)
# Reimplementation of MultioutSparseGPModel but I code inference myself,
# inspired from GPy and not relying on GPyTorch
# Same structure as MutlitoutExactGPModel but with GPy sparse inference,
# i.e. different forward function

class My_MultioutSparseGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel, inducing_points=torch.tensor([]), output_size=1):
        assert kernel.batch_shape[0] != 0, \
            'The kernel and likelihood have not been given the batch_shape ' \
            'parameter to indicate the number of tasks.'
        self.output_size = output_size
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        super().__init__(train_x, train_y, likelihood, kernel)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([output_size]))

        self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        self.register_added_loss_term("inducing_point_loss_term")
        # As in GPy VarDTC inference
        # https://gpy.readthedocs.io/en/deploy/_modules/GPy/inference/latent_function_inference/var_dtc.html#VarDTC
        # TODO compute all things needed for inference in init, then call in
        #  forward

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Standard GPyTorch: calls Kernel(nn.Module).__call__(x), which calls
        # lazy(InducingKernel(Kernel).forward(x1,x2 both = x))
        # Standard GPy: call model.predict(x) which calls
        # self.posterior._raw_predict(x) which comes from
        # self.inference_method.inference(x), which computes & caches on train

        multitask_result = \
            gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
        return multitask_result

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            observed_mean = observed_pred.mean
            observed_var = observed_pred.variance
            # Workaround in multi-output case, might be a bit slow...
            mean = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                           observed_mean.shape[1]]))
            var = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                          observed_var.shape[1]]))
            for i in range(observed_mean.shape[1]):
                mean[:, :, i] = torch.autograd.grad(
                    observed_mean[:, i].sum(), test_x, retain_graph=True)[0]
                if not onlymean:
                    var[:, :, i] = torch.autograd.grad(
                        observed_var[:, i].sum(), test_x, retain_graph=True)[0]
        return mean.transpose(1, 2), var.transpose(1, 2)  # (N, dy, dx)


# SVGP for single output GP
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
# https://docs.gpytorch.ai/en/stable/variational.html
# https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
# SVGP uses minibatches of data to speed up training! Overall it needs
# better/longer optimization than SGPR, so the number of iterations and
# initialization become more important. But minibatch training helps reduce
# the training time (and memory), and prediction is faster.

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, kernel,
                 inducing_points=torch.tensor([]), minibatch_size=124,
                 learn_inducing_locations=True,
                 variational_distribution=gpytorch.variational.CholeskyVariationalDistribution,
                 variational_strategy=gpytorch.variational.VariationalStrategy):
        train_x = torch.squeeze(train_x)
        train_y = torch.squeeze(train_y)
        train_y = train_y.contiguous()
        self.minibatch_size = minibatch_size
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        variational_distribution = \
            variational_distribution(inducing_points.size(0))
        variational_strategy = variational_strategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.variational_strategy = variational_strategy
        self.variational_distribution = variational_distribution
        if 'Natural' in self.variational_distribution.__class__.__name__:
            self.NGD = True
        else:
            self.NGD = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, likelihood, train_x, train_y, training_iter=50,
                 optimizer=torch.optim.Adam, lr=0.1, optim_options={},
                 scheduler=None, scheduler_options={}, stopper=None,
                 parameters=None, verbose=False):
        train_x = torch.squeeze(train_x).double()
        train_y = torch.squeeze(train_y).double().contiguous()
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.minibatch_size, shuffle=True)
        # https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html
        losses = []
        self.train()
        likelihood.train()
        if 'variational_ngd_lr' in optim_options.keys():
            variational_ngd_lr = optim_options['variational_ngd_lr']
            optim_options = optim_options.copy()
            optim_options.pop('variational_ngd_lr')
        else:
            variational_ngd_lr = 0.1
        if self.NGD:
            # https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html
            if not parameters:
                variational_ngd_optimizer = gpytorch.optim.NGD(
                    self.variational_parameters(), num_data=train_y.size(0),
                    lr=variational_ngd_lr)
                hyperparameters = [{'params': self.hyperparameters()},
                                   {'params': likelihood.parameters()}]
            else:
                variational_ngd_optimizer = gpytorch.optim.NGD(
                    self.variational_parameters(), num_data=train_y.size(0),
                    lr=variational_ngd_lr)
                # hyperparameters = parameters - self.variational_parameters()
                hyperparameters = set(parameters)
                for name, param in self.named_variational_parameters():
                    if param in hyperparameters:
                        hyperparameters -= {param}
                hyperparameters = list(hyperparameters)
                logging.warning('Parameters were given with Natural Gradient '
                                'Descent: variational parameters need to be '
                                'handled by NGD only, so we take out the '
                                'variational parameters from the given '
                                'parameters, and only optimize the rest with '
                                'SGD.')
        if 'LBFGS' in optimizer.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('LBFGS is slow and memory intensive but also '
                            'converges well (with strong_wolfe). Also, it can '
                            'only handle model parameters (not likelihood). It '
                            'does not support gain scheduling and early '
                            'stopping!')
            if self.NGD:
                if not parameters:
                    hyperparameters = hyperparameters[:-1]
            elif not parameters:
                parameters = [{'params': self.parameters()}]
            if stopper or scheduler:
                raise NotImplementedError
        if self.NGD:
            parameters = hyperparameters
        elif not parameters:
            parameters = [{'params': self.parameters()},
                          {'params': likelihood.parameters()}]
        if 'mll' in optim_options.keys():
            mll = optim_options['mll']
            optim_options = optim_options.copy()
            optim_options.pop('mll')
        else:
            mll = gpytorch.mlls.VariationalELBO
        optimizer = optimizer(parameters, lr=lr, **optim_options)
        if scheduler:
            scheduler = scheduler(optimizer, verbose=False,
                                  **scheduler_options)
        mll = mll(likelihood, self, num_data=train_y.shape[0])
        epochs_iter = tqdm.tqdm(range(training_iter), desc="Epoch", leave=True)
        for i in epochs_iter:
            start = time.time()
            # Within each epoch, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch",
                                       leave=False)  # nb depends size
            for x_batch, y_batch in minibatch_iter:
                if self.NGD:
                    if 'LBFGS' in optimizer.__class__.__name__:
                        def closure():
                            if torch.is_grad_enabled():
                                variational_ngd_optimizer.zero_grad()
                                optimizer.zero_grad()
                            output = self(x_batch)
                            loss = -mll(output, y_batch)
                            losses.append(loss.item())
                            minibatch_iter.set_postfix(loss=loss.item())
                            if loss.requires_grad:
                                loss.backward()
                            return loss

                        variational_ngd_optimizer.step()
                        optimizer.step(closure)
                        loss = losses[-1]
                    else:
                        # Perform NGD step to optimize variational parameters
                        variational_ngd_optimizer.zero_grad()
                        optimizer.zero_grad()
                        output = self(x_batch)
                        loss = -mll(output, y_batch)
                        losses.append(loss.item())
                        minibatch_iter.set_postfix(loss=loss.item())
                        loss.backward()
                        variational_ngd_optimizer.step()
                        optimizer.step()
                else:
                    if 'LBFGS' in optimizer.__class__.__name__:
                        def closure():
                            if torch.is_grad_enabled():
                                optimizer.zero_grad()
                            output = self(x_batch)
                            loss = -mll(output, y_batch)
                            losses.append(loss.item())
                            minibatch_iter.set_postfix(loss=loss.item())
                            if loss.requires_grad:
                                loss.backward()
                            return loss

                        optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        output = self(x_batch)
                        loss = -mll(output, y_batch)
                        losses.append(loss.item())
                        minibatch_iter.set_postfix(loss=loss.item())
                        loss.backward()
                        optimizer.step()
            end = time.time()
            # Within each epoch, update learning rate and check early stopping
            if 'LBFGS' in optimizer.__class__.__name__:
                if verbose:
                    print('Iter %d/%d - Loss' % (i + 1, training_iter),
                          'in ', str(end - start))
            else:
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()), 'in',
                          str(end - start))
                epochs_iter.set_postfix(loss=loss.item())
                if scheduler:
                    scheduler.step(loss)
                if stopper:
                    stopper(loss)
                    if stopper.early_stop:
                        # Stop early
                        break  # TODO in minibatch?
        return losses

    def predict(self, x, likelihood, full_cov=False, minibatch_size=None,
                onlymean=False):
        if minibatch_size:
            # Could also pass minibatches of x for prediction if necessary
            # https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
            logging.error('Not implemented yet.')
        self.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/728
            observed_pred = likelihood(self(x))
            if not full_cov:
                mean = observed_pred.mean
                if onlymean:
                    var = torch.zeros_like(mean, device=mean.device)
                else:
                    var = observed_pred.variance
            else:
                mean = observed_pred.mean
                if onlymean:
                    var = torch.zeros((mean.shape[1], mean.shape[1]),
                                      device=mean.device)
                else:
                    var = observed_pred.covariance_matrix
        # Reshape: if dx or dy = 1 then output will otherwise be vector!
        if len(x) == 1:
            mean = reshape_pt1(mean)
            var = reshape_pt1(var)
        elif len(x.shape) > 1:
            if len(mean.shape) == 1:
                mean = reshape_dim1(mean)
                var = reshape_dim1(var)
        return mean, var

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            mean = torch.autograd.grad(observed_pred.mean.sum(), test_x,
                                       retain_graph=True)[0]
            if onlymean:
                var = torch.zeros_like(mean)
            else:
                var = torch.autograd.grad(observed_pred.variance.sum(), test_x,
                                          retain_graph=True)[0]
        # Reshape: if dx = 1 then output will otherwise be vector!
        if len(x) == 1:
            mean = reshape_pt1(mean)
            var = reshape_pt1(var)
        elif len(x.shape) > 1:
            if len(mean.shape) == 1:
                mean = reshape_dim1(mean)
                var = reshape_dim1(var)
        return mean, var  # (N, dx)


# SVGP for independent multiple outputs (multiout)
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html

class MultioutSVGPModel(SVGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel,
                 inducing_points=torch.tensor([]), output_size=1,
                 minibatch_size=124, share_inducing_points=False,
                 learn_inducing_locations=True,
                 variational_distribution=gpytorch.variational.CholeskyVariationalDistribution,
                 base_variational_strategy=gpytorch.variational.VariationalStrategy):
        assert kernel.batch_shape[0] != 0, \
            'The kernel and likelihood have not been given the batch_shape ' \
            'parameter to indicate the number of tasks.'
        train_x = torch.squeeze(train_x)
        train_y = torch.squeeze(train_y)
        train_y = train_y.contiguous()
        self.minibatch_size = minibatch_size
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        if not share_inducing_points:
            # One set of inducing points for each output dim
            inducing_points = inducing_points.repeat(output_size, 1, 1)
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = variational_distribution(
            inducing_points.size(-2), batch_shape=torch.Size([output_size]))
        variational_strategy = \
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=learn_inducing_locations),
                num_tasks=output_size)
        gpytorch.models.ApproximateGP.__init__(self, variational_strategy)
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([output_size]))
        self.covar_module = kernel
        self.variational_strategy = variational_strategy
        self.variational_distribution = variational_distribution
        if 'Natural' in self.variational_distribution.__class__.__name__:
            self.NGD = True
        else:
            self.NGD = False

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            observed_mean = observed_pred.mean
            observed_var = observed_pred.variance
            # Workaround in multi-output case, might be a bit slow...
            mean = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                           observed_mean.shape[1]]))
            var = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                          observed_var.shape[1]]))
            for i in range(observed_mean.shape[1]):
                mean[:, :, i] = torch.autograd.grad(
                    observed_mean[:, i].sum(), test_x, retain_graph=True)[0]
                if not onlymean:
                    var[:, :, i] = torch.autograd.grad(
                        observed_var[:, i].sum(), test_x, retain_graph=True)[0]
        return mean.transpose(1, 2), var.transpose(1, 2)  # (N, dy, dx)


# SVGP for correlated multiple outputs (multitask)
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html

class MultitaskSVGPModel(SVGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel, num_latents,
                 inducing_points=torch.tensor([]), output_size=1,
                 minibatch_size=124, share_inducing_points=False,
                 learn_inducing_locations=True,
                 variational_distribution=gpytorch.variational.CholeskyVariationalDistribution,
                 base_variational_strategy=gpytorch.variational.VariationalStrategy):
        assert kernel.batch_shape[0] != 0, \
            'The kernel and likelihood have not been given the batch_shape ' \
            'parameter to indicate the number of tasks.'
        train_x = torch.squeeze(train_x)
        train_y = torch.squeeze(train_y)
        train_y = train_y.contiguous()
        self.minibatch_size = minibatch_size
        if len(inducing_points) == 0:
            nb_inducing_points = int(np.floor(len(train_x) / 10))
            random_idx = torch.randperm(len(train_x))[:nb_inducing_points]
            inducing_points = train_x[random_idx]
        if not share_inducing_points:
            # One set of inducing points for each output dim
            inducing_points = inducing_points.repeat(num_latents, 1, 1)
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = variational_distribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a
        # batch output
        variational_strategy = \
            gpytorch.variational.LMCVariationalStrategy(
                base_variational_strategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=learn_inducing_locations),
                num_tasks=output_size,
                num_latents=num_latents,
                latent_dim=-1)
        gpytorch.models.ApproximateGP.__init__(self, variational_strategy)
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]))
        self.covar_module = kernel
        self.variational_strategy = variational_strategy
        self.variational_distribution = variational_distribution
        if 'Natural' in self.variational_distribution.__class__.__name__:
            self.NGD = True
        else:
            self.NGD = False

    def predict_gradient(self, x, likelihood, onlymean=False):
        self.eval()
        likelihood.eval()
        test_x = x.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/772
            # https://github.com/cornellius-gp/gpytorch/issues/1116
            observed_pred = likelihood(self(test_x))
            observed_mean = observed_pred.mean
            observed_var = observed_pred.variance
            # Workaround in multi-output case, might be a bit slow...
            mean = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                           observed_mean.shape[1]]))
            var = torch.zeros(torch.Size([len(test_x), test_x.shape[1],
                                          observed_var.shape[1]]))
            for i in range(observed_mean.shape[1]):
                mean[:, :, i] = torch.autograd.grad(
                    observed_mean[:, i].sum(), test_x, retain_graph=True)[0]
                if not onlymean:
                    var[:, :, i] = torch.autograd.grad(
                        observed_var[:, i].sum(), test_x, retain_graph=True)[0]
        return mean.transpose(1, 2), var.transpose(1, 2)  # (N, dy, dx)


# Multi ouput GPs with different kernels and likelihoods for each
# independent output dimension BUT seems to have lower performance than
# previous one in general!!
# https://docs.gpytorch.ai/en/v1.1.1/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html

class ModelListGPModel(gpytorch.models.IndependentModelList):
    def __init__(self, train_x, train_y, likelihoods, kernels, output_size=1):
        # Create one ExactGPModel per output dimension
        train_y = reshape_dim1(train_y)
        train_y = train_y.contiguous()
        model_list = []
        for i in range(output_size):
            if (type(likelihoods) is list) and not (type(kernels) is list):
                model = ExactGPModel(train_x, train_y[:, i], likelihoods[i],
                                     kernels)
            elif not (type(likelihoods) is list) and (type(kernels) is list):
                model = ExactGPModel(train_x, train_y[:, i], likelihoods,
                                     kernels[i])
            elif (type(likelihoods) is list) and (type(kernels) is list):
                model = ExactGPModel(train_x, train_y[:, i], likelihoods[i],
                                     kernels[i])
            else:
                model = ExactGPModel(train_x, train_y[:, i], likelihoods,
                                     kernels)
            model_list += [model]
        super().__init__(*model_list)
        likelihood_list = [model.likelihood for model in model_list]
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)
        self.output_size = output_size

    def optimize(self, likelihood, train_x, train_y, training_iter=50,
                 optimizer=torch.optim.Adam, lr=0.1, optim_options={},
                 scheduler=None, scheduler_options={}, stopper=None,
                 parameters=None, verbose=False):
        # https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html
        losses = []
        self.train()
        likelihood.train()
        if 'LBFGS' in optimizer.__name__:
            # http://sagecal.sourceforge.net/pytorch/index.html
            logging.warning('LBFGS is slow and memory intensive but '
                            'also converges well (with strong_wolfe). It does '
                            'not support gain scheduling and early stopping!')
            if stopper or scheduler:
                raise NotImplementedError
        if not parameters:
            parameters = self.parameters()
        if 'mll' in optim_options.keys():
            mll = optim_options['mll']
            optim_options = optim_options.copy()
            optim_options.pop('mll')
        else:
            mll = gpytorch.mlls.SumMarginalLogLikelihood
        optimizer = optimizer(parameters, lr=lr, **optim_options)
        if scheduler:
            scheduler = scheduler(optimizer, verbose=verbose,
                                  **scheduler_options)
        mll = mll(likelihood, self)
        for i in range(training_iter):
            if 'LBFGS' in optimizer.__class__.__name__:
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    output = self(*self.train_inputs)
                    loss = -mll(output, self.train_targets)
                    losses.append(loss.item())
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                start = time.time()
                optimizer.step(closure)
                end = time.time()
                if verbose:
                    print('Iter %d/%d - Loss' % (i + 1, training_iter), 'in ',
                          str(end - start))
            else:
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(*self.train_inputs)
                # Calculate loss and backpropagate gradients
                loss = -mll(output, self.train_targets)
                losses.append(loss.item())
                loss.backward()
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()))
                optimizer.step()
                if scheduler:
                    scheduler.step(loss)
                if stopper:
                    stopper(loss)
                    if stopper.early_stop:
                        # Stop early
                        break
        return losses

    def predict(self, x_list, likelihood, full_cov=False, onlymean=False):
        self.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # https://github.com/cornellius-gp/gpytorch/issues/728
            predictions = likelihood(*self(*x_list))
            mean = []
            var = []
            for submodel, prediction in zip(self.models, predictions):
                current_mean = prediction.mean
                if not full_cov:
                    if onlymean:
                        current_var = torch.zeros_like(current_mean)
                    else:
                        current_var = observed_pred.variance
                else:
                    if onlymean:
                        current_var = torch.zeros((
                            current_mean.shape[1], current_mean.shape[1]),
                            device=current_mean.device)
                    else:
                        current_var = observed_pred.covariance_matrix
                if len(x_list[0]) == 1:
                    current_mean = reshape_pt1(current_mean)
                    current_var = reshape_pt1(current_var)
                elif len(x_list[0].shape) > 1:
                    if len(current_mean.shape) == 1:
                        current_mean = reshape_dim1(current_mean)
                        current_var = reshape_dim1(current_var)
                mean += [current_mean]
                var += [current_var]
            return mean, var
