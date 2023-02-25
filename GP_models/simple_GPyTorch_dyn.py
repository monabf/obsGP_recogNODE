import copy
import logging
import os
import sys
import time

import GPy
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sb
import torch
import dill as pkl
from matplotlib import pyplot as plt
from itertools import chain

from GP_models.GPyTorch_models import ExactGPModel, MultioutGPModel, \
    SparseGPModel, MultioutSparseGPModel, SVGPModel, MultioutSVGPModel, \
    MultitaskSVGPModel
from utils.pytorch_utils import StandardScaler, print_parameters
from utils.config import Config
from simulation.controllers import sin_controller_1D, sin_controller_02D, \
    null_controller
from model_evaluation.plotting_closedloop_rollouts import \
    plot_closedloop_rollout_data, save_closedloop_rollout_variables
from model_evaluation.plotting_functions import plot_model_evaluation, \
    save_GP_data, plot_GP, run_rollouts
from model_evaluation.plotting_kalman_rollouts import \
    save_kalman_rollout_variables, plot_kalman_rollout_data
from model_evaluation.plotting_rollouts import plot_rollout_data, \
    save_rollout_variables
from utils.Keops_utils import kMeans
from utils.utils import reshape_pt1, remove_outlier, RMS, save_log, \
    log_multivariate_normal_likelihood, reshape_dim1, concatenate_lists

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
sb.set_style('whitegrid')
plot_params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif': 'Palatino',
    'font.size': 16,
    "pgf.preamble": "\n".join([
        r'\usepackage{bm}',
    ]),
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage{amssymb}',
                            r'\usepackage{cmbright}'],
}
plt.rcParams.update(plot_params)

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Class to learn simple dynamics GP from mappings (X,U)=(x_t, u_t) -> Y=(y_t)
# or (x_t+1), that already contain noise chosen by the user. Takes care of
# evaluating/saving GP and other utilities.
# For autonomous system where U=0 uniformly: make U a matrix of zeros,
# control ignored for learning

# With pytorch: use .clone() instead of .copy(), slightly different behavior so
# really use as soon as need copy just for size! Odeint can only have outputs
# of same size as inputs, cut only after. GPyTorch tends to squeeze matrices,
# so explicitly reshape as necessary. ARD kernel = dx lengthscales, multitask
# GP = dy kernels, each with dx lengthscales, 1 outputscale, 1 noise
# variance. Give the kernel either dx x dy or only dx prior lengthscales,
# either 1 or dy outputscales, and the likelihood either 1 (if Gaussian) or dy
# (if mMultitaskGaussian) noise variances (both in likelihood.noise and
# likelihood.noise_covar.noise). Do not store kernel objects inside lists,
# tuples, or dictionaries, but as class properties or in torch.nn.ModuleList
# (Lucas). Pytorch differentiates between double/float/long etc, so make sure
# you use double by default and write tensor(1.). Pay attention to the
# definition of likelihood, kernel and GP model class, most errors there!

LOCAL_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
LOCAL_PATH_TO_SRC = LOCAL_PATH.split(os.sep + 'src', 1)[0]


class Simple_GPyTorch_Dyn:

    def __getattr__(self, item):
        # self.config[item] can be called directly as self.item
        # https://stackoverflow.com/questions/42272335/how-to-make-a-class-which-has-getattr-properly-pickable
        try:
            return self.__dict__[item]
        except KeyError:
            try:
                return self.config[item]
            except KeyError:
                if item.startswith('__') and item.endswith('__'):
                    raise AttributeError(item)
                else:
                    logging.info(f'No item {item}')
                    return None

    def __getstate__(self):
        # needed for pickle with custom __getattr__:
        # https://stackoverflow.com/questions/50888391/pickle-of-object-with-getattr-method-in-python-returns-typeerror-object-no
        if self.use_GPy:
            # to handle pickling exception with GPy kernels
            # https://github.com/SheffieldML/GPy/issues/605
            # https://stackoverflow.com/questions/2999638/how-to-stop-attributes-from-being-pickled-in-python
            # https://stackoverflow.com/questions/3405715/elegant-way-to-remove-fields-from-nested-dictionaries
            d = {}
            avoid_keys = ['kernel']
            for k, v in self.__dict__.items():
                if k not in avoid_keys:
                    if isinstance(v, dict):
                        d[k] = {}
                        for kp, vp in v.items():
                            if kp not in avoid_keys:
                                d[k][kp] = vp
                    else:
                        d[k] = v
            return d
        else:
            return self.__dict__

    def __setstate__(self, d):
        # needed for pickle with custom __getattr__:
        # https://stackoverflow.com/questions/50888391/pickle-of-object-with-getattr-method-in-python-returns-typeerror-object-no
        # self.__dict__.update(d)
        # to handle pickling exception with used GPy kernels
        # https://github.com/SheffieldML/GPy/issues/605
        # https://stackoverflow.com/questions/2999638/how-to-stop-attributes-from-being-pickled-in-python
        logging.warning('If you are loading this model from pickle: it can '
                        'only be serialized once, can only be used for '
                        'prediction on the GPu if it was trained on the GPU, '
                        'and if it contains a GPy model then you need to set a '
                        'new kernel because GPy kernels are not serializable.')
        self.__dict__.update(d)

    def __init__(self, X, U, Y, config: Config, ground_truth_approx=False):
        assert len(X) == len(U) == len(Y), 'X, U and Y must have the same ' \
                                           'length'
        self.config = config
        self.specs = self.config
        self.X = reshape_pt1(X)
        self.U = reshape_pt1(U)
        self.Y = reshape_pt1(Y)
        if torch.cuda.is_available() and not self.use_GPy:
            self.X, self.U, self.Y = self.X.cuda(), self.U.cuda(), self.Y.cuda()
        logging.warning(
            'Using PyTorch: all data used by the GP framework must be torch '
            'tensors and not numpy data. Conversions should be avoided at '
            'best. Hence, all data inside and outside of GP should be pytorch '
            'as possible, for readability and to avoid converting back and '
            'forth from GPU to CPU. Only functions that convert to numpy are '
            'for saving data to csv and plots.')
        self.device = self.X.device

        if self.use_GPy:
            logging.warning(
                'Using GPy for the model in PyTorch code!! All GPy code is on '
                'the CPU, whereas the PyTorch parts can be done on the GPU. '
                'Therefore, avoid using the GPU for the GP part, use it only '
                'outside of this class if necessary. (Quick and dirty fix to '
                'be able to use GPy SparseGP)')
            logging.getLogger("paramz").setLevel(logging.ERROR)

        if self.sliding_window_size_each:
            self.X = self.X[-self.sliding_window_size_each:, :]
            self.U = self.U[-self.sliding_window_size_each:, :]
            self.Y = self.Y[-self.sliding_window_size_each:, :]

        if self.sliding_window_size_each and self.sliding_window_size:
            logging.warning('Both a sliding window over the whole dataset and '
                            'one over each training set have been defined, '
                            'make sure this is really what you want.')

        if self.sparse and self.sparse.get('method') == 'kMeans_each':
            # Sparsify each subdataset with kMeans clustering, then ExactGP
            # Work on points (x(t), u(t), x(t+dt))
            D = torch.cat((self.X, self.U, self.Y), dim=1)
            k = self.sparse.get('nb_inducing_inputs')
            cl, c = kMeans(D, k, **self.GP_model_options)
            # Filter nans out
            c = c[~torch.any(c.isnan(), dim=1)]
            self.X = c[:, :self.X.shape[1]]
            self.U = c[:, self.X.shape[1]:self.X.shape[1] + self.U.shape[1]]
            self.Y = c[:, self.X.shape[1] + self.U.shape[1]:]

        if (self.Y.shape[1] > 1) and not self.multitask_GP:
            # Default for multiple outputs are batch independent outputs,
            # use multitask GP (= correlated outputs) only if specified
            self.multioutput_GP = True

        if self.dt > 0.1:
            logging.warning('Time step is larger than 0.1s! This might be too '
                            'much, most of all for all small operations that '
                            'rely on Euler discretization to obtain continuous '
                            'solutions from discrete GP models, such as '
                            'rollouts, continuous observers...')
        if ((X.shape[1] != Y.shape[1]) and (self.nb_rollouts > 0)) and not \
                (('Michelangelo' in self.system) or (
                        'justvelocity' in self.system)):
            raise ValueError('The rollout function is only available for '
                             'dynamics GPs learning x_t -> x_t+1, x_t -> '
                             'velocity_t+1, x_t -> velocitydot_t, '
                             'or in particular cases where it has been '
                             'precoded. To learn another type of GP '
                             'x_t -> y_t, set the number of rollouts to zero.')
        if self.hyperparam_optim:
            assert any(self.hyperparam_optim == k for k in (
                'fixed_start', 'fixed_hyperparameters', 'only_likelihood',
                'fixed_all_parameters')), \
                'Only possible options for hyperparam_optim are None, ' \
                'fixed_start, only_likelihood, fixed_hyperparameters and ' \
                'all_fixed_params.'
        if self.sparse:
            assert isinstance(self.sparse, dict), \
                'The sparse parameter must be a dictionary containing the ' \
                'method (for now SparseGP, SVGP, kMeans, kMeans_each) and ' \
                'the number of inducing inputs nb_inducing_inputs, which are ' \
                'otherwise set to default.'
        if not self.grid_inf:
            logging.warning('No grid was predefined by the user for one step '
                            'ahead model evaluation and rollouts, so using '
                            'min and max of state data.')
            self.grid_inf = torch.min(self.X, dim=0).values
            self.grid_sup = torch.max(self.X, dim=0).values

        self.step = 0
        self.sample_idx = 0
        if ground_truth_approx:
            # Data rollouts cannot be longer than data
            self.rollout_length = int(
                np.min([self.rollout_length, len(self.X) - 1]))
        self.prior_kwargs = self.config.prior_kwargs
        self.ground_truth_approx = ground_truth_approx
        if self.ground_truth_approx:
            logging.warning('True dynamics are approximated from data or '
                            'from a simplified model: there is actually no '
                            'ground truth, the true dynamics are only used as '
                            'a comparison to the GP model! Hence, model '
                            'evaluation tools such as GP_plot, rollouts or '
                            'model_evaluation are only indicative; true '
                            'evaluation of the model can only be obtained by '
                            'predicting on a test set and comparing to the '
                            'true data.')
        self.hyperparams = np.array([[]])
        # Metrics are important to monitor learning! See
        # https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
        # and GPML, Rasmussen, Section 2.5
        self.RMSE_time = torch.zeros((0, 2))
        self.SRMSE_time = torch.zeros((0, 2))
        self.log_likelihood_time = torch.zeros((0, 2))
        self.stand_log_likelihood_time = torch.zeros((0, 2))
        self.stop = False
        self.current_time = time.time() - time.time()
        self.time = reshape_pt1(torch.tensor([[0, 0]]))
        self.observer_RMSE = torch.zeros((0, 2))
        self.observer_SRMSE = torch.zeros((0, 2))
        self.output_RMSE = torch.zeros((0, 2))
        self.output_SRMSE = torch.zeros((0, 2))
        self.rollout_RMSE = torch.zeros((0, 2))
        self.rollout_SRMSE = torch.zeros((0, 2))
        self.rollout_log_AL = torch.zeros((0, 2))
        self.rollout_stand_log_AL = torch.zeros((0, 2))
        self.kalman_rollout_RMSE = torch.zeros((0, 2))
        self.kalman_rollout_SRMSE = torch.zeros((0, 2))
        self.kalman_rollout_log_AL = torch.zeros((0, 2))
        self.kalman_rollout_stand_log_AL = torch.zeros((0, 2))
        self.closedloop_rollout_RMSE = torch.zeros((0, 2))
        self.closedloop_rollout_SRMSE = torch.zeros((0, 2))
        self.closedloop_rollout_log_AL = torch.zeros((0, 2))
        self.closedloop_rollout_stand_log_AL = torch.zeros((0, 2))

        # Create grid of (x_t, u_t) to evaluate the GP quality (true dynamics
        # needed to compare true x_t+1 to predicted)
        self.grid, self.grid_controls = self.create_grid(self.init_state,
                                                         self.init_control,
                                                         self.constrain_u,
                                                         self.grid_inf,
                                                         self.grid_sup)
        true_predicted_grid = \
            self.create_true_predicted_grid(self.grid, self.grid_controls)
        self.true_predicted_grid = \
            true_predicted_grid.reshape(-1, self.Y.shape[1])
        # Reject outliers from grid
        true_predicted_grid_df = pd.DataFrame(
            self.true_predicted_grid.cpu().numpy())
        grid_df = pd.DataFrame(self.grid.cpu().numpy())
        grid_controls_df = pd.DataFrame(self.grid_controls.cpu().numpy())
        mask = remove_outlier(true_predicted_grid_df)
        true_predicted_grid_df = true_predicted_grid_df[mask]
        grid_df = grid_df[mask]
        grid_controls_df = grid_controls_df[mask]
        self.true_predicted_grid = \
            true_predicted_grid_df.values.reshape(-1, self.Y.shape[1])
        self.grid = grid_df.values.reshape(-1, self.X.shape[1])
        self.grid_controls = \
            grid_controls_df.values.reshape(-1, self.U.shape[1])
        self.rollout_list = self.create_rollout_list()
        self.variables = {'X': self.X, 'Y': self.Y, 'Computation_time':
            self.time}
        self.variables['RMSE_time'] = self.RMSE_time
        self.variables['SRMSE_time'] = self.SRMSE_time
        self.variables['log_AL_time'] = self.log_likelihood_time
        self.variables['stand_log_AL_time'] = self.stand_log_likelihood_time

        # Create unique results folder
        params = str(np.random.uniform()) + '_' + str(
            self.nb_samples) + 'samples_' + 't' + str(
            self.specs.get('t0')) + '-' + str(self.specs.get('tf')) + '_' + str(
            self.true_meas_noise_var) + 'meas_' + str(
            self.process_noise_var) + 'process'
        if self.restart_on_loop:
            params = 'Restarts_' + params
        if self.sparse:
            params = self.sparse.get('method') + '_' + params
        if self.use_GPy:
            params = 'GPy_' + params
        if not self.constrain_u:
            params += '_unclipped'
        else:
            params += '_uclip' + str(np.max(self.constrain_u))
        if self.nb_loops > 1:
            self.results_folder = os.path.join(str(LOCAL_PATH_TO_SRC),
                                               'Figures', str(self.system),
                                               str(self.nb_loops) + '_pass',
                                               str(self.nb_rollouts) +
                                               '_rollouts', params, 'Loop_0')
        else:
            self.results_folder = os.path.join(
                str(LOCAL_PATH_TO_SRC), 'Figures', str(self.system),
                'Single_pass', str(self.nb_rollouts) + '_rollouts', params)
        if self.__class__.__name__ == 'Simple_GPyTorch_Dyn':
            # Only do if constructor not called from inherited class
            os.makedirs(self.results_folder, exist_ok=False)
            self.save_grid_variables(self.grid, self.grid_controls,
                                     self.true_predicted_grid,
                                     self.results_folder)
            self.true_predicted_grid = torch.as_tensor(
                self.true_predicted_grid, device=self.device)
            self.grid = torch.as_tensor(self.grid, device=self.device)
            self.grid_controls = torch.as_tensor(self.grid_controls,
                                                 device=self.device)
            self.grid_variables = {'Evaluation_grid': self.grid,
                                   'Grid_controls': self.grid_controls}
            save_rollout_variables(self, self.results_folder, self.nb_rollouts,
                                   self.rollout_list, step=self.step,
                                   ground_truth_approx=self.ground_truth_approx,
                                   plots=self.monitor_experiment)
            # Save log in results folder
            os.rename(str(LOCAL_PATH_TO_SRC) + '/Figures/Logs/' + 'log' +
                      str(sys.argv[1]) + '.log',
                      os.path.join(self.results_folder,
                                   'log' + str(sys.argv[1]) + '.log'))
            save_log(self.results_folder)
            if self.verbose:
                logging.info(self.results_folder)

    def learn(self, new_X=[], new_U=[], new_Y=[]):
        self.step += 1
        if self.verbose:
            logging.info('Update GP for the ' + str(self.step) + 'th time')
        # logging.getLogger("paramz").setLevel(logging.ERROR)

        # Update data, model and hyperparameters
        self.update_data(new_X=new_X, new_U=new_U, new_Y=new_Y)
        start = time.time()
        self.update_model()
        # Record computation time for learning (not for other tasks)
        self.current_time += time.time() - start
        self.time = torch.cat((self.time, reshape_pt1(
            torch.tensor([torch.tensor(self.sample_idx), self.current_time]))),
                              dim=0)
        return self.model

    def create_GP_model(self):
        # Create GP model
        if self.use_GPy:
            if self.sparse:
                # https://github.com/SheffieldML/GPy/issues/602
                if self.sparse.get('nb_inducing_inputs'):
                    self.nb_inducing_inputs = np.min([self.sparse.get(
                        'nb_inducing_inputs'), len(self.GP_X)])
                else:
                    self.nb_inducing_inputs = int(np.floor(len(self.GP_X) / 10))
                    self.sparse.update({'default_nb_inducing_inputs': 'len(X)/10'})
                random_idx = np.random.choice(len(self.GP_X),
                                              self.nb_inducing_inputs,
                                              replace=False)
                if self.nb_inducing_inputs == 1:
                    Z = reshape_pt1(self.GP_X[random_idx])
                else:
                    Z = reshape_dim1(self.GP_X[random_idx])
                self.variables['GP_inducing_inputs'] = Z
                if self.nb_inducing_inputs >= len(self.GP_X):
                    logging.warning('More inducing points than data, using '
                                    'regular GP instead of sparse approximation.')
                    self.model = GPy.core.gp.GP(self.GP_X.numpy(),
                                                self.GP_Y.numpy(),
                                                kernel=self.kernel,
                                                likelihood=GPy.likelihoods.Gaussian(
                                                    variance=self.meas_noise_var),
                                                inference_method=GPy.inference.latent_function_inference.ExactGaussianInference())
                elif (self.sparse.get('method') == 'VarDTC') or \
                        not (self.sparse.get('method')):
                    self.model = GPy.core.SparseGP(self.GP_X.numpy(),
                                                   self.GP_Y.numpy(),
                                                   Z.numpy(),
                                                   kernel=self.kernel,
                                                   likelihood=GPy.likelihoods.Gaussian(
                                                       variance=self.meas_noise_var),
                                                   inference_method=GPy.inference.latent_function_inference.VarDTC()
                                                   )
                else:
                    raise NotImplementedError('This sparsification method is '
                                              'not implemented, use VarDTC.')
                for i in range(len(self.model.kern.parameters)):
                    self.model.kern.parameters[i].constrain_bounded(1e-3, 200.)
            else:
                self.model = GPy.core.gp.GP(self.GP_X.numpy(),
                                            self.GP_Y.numpy(),
                                            kernel=self.kernel,
                                            likelihood=GPy.likelihoods.Gaussian(
                                                variance=self.meas_noise_var),
                                            inference_method=GPy.inference.latent_function_inference.ExactGaussianInference())
                for i in range(len(self.model.kern.parameters)):
                    self.model.kern.parameters[i].constrain_bounded(1e-3, 200.)
            self.model.preferred_optimizer = 'lbfgsb'
            if self.hyperparam_optim and self.step == 1:
                # Fix the starting point for hyperparameter optimization to the
                # parameters at initialization
                if self.hyperparam_optim == 'fixed_start':
                    self.start_optim_fixed = self.model.optimizer_array.copy()
            return 'GPy model'

        if self.GP_model_options:
            GP_model_options = self.GP_model_options
        else:
            GP_model_options = {}
        if not self.likelihood:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.multioutput_GP and self.sparse:
            logging.warning('MultitaskGaussianLikelihood needed for '
                            'multioutput GPs')
            if self.sparse.get('nb_inducing_inputs'):
                self.nb_inducing_inputs = np.min([self.sparse.get(
                    'nb_inducing_inputs'), len(self.GP_X)])
            else:
                self.nb_inducing_inputs = int(np.floor(len(self.GP_X) / 10))
                self.sparse.update({'default_nb_inducing_inputs': 'len(X)/10'})
            random_idx = \
                torch.randperm(len(self.GP_X))[:self.nb_inducing_inputs]
            if self.nb_inducing_inputs == 1:
                self.GP_inducing_inputs = reshape_pt1(self.GP_X[random_idx])
            else:
                self.GP_inducing_inputs = reshape_dim1(self.GP_X[random_idx])


            self.variables['GP_inducing_inputs'] = self.GP_inducing_inputs
            if self.sparse.get('method') == 'SparseGP':
                self.model = MultioutSparseGPModel(
                    self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                    inducing_points=self.GP_inducing_inputs,
                    output_size=self.GP_Y.shape[1])
            elif self.sparse.get('method') == 'SVGP':
                self.model = MultioutSVGPModel(
                    self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                    inducing_points=self.GP_inducing_inputs,
                    output_size=self.GP_Y.shape[1],
                    minibatch_size=self.minibatch_size, **GP_model_options)
            elif 'kMeans' in self.sparse.get('method'):
                self.model = MultioutGPModel(
                    self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                    output_size=self.GP_Y.shape[1])
            else:
                logging.error('This sparse method is not implemented')
                raise NotImplementedError
        elif self.multitask_GP and self.sparse:
            logging.warning('MultitaskGaussianLikelihood needed for '
                            'multitask GPs')
            if self.sparse.get('nb_inducing_inputs'):
                self.nb_inducing_inputs = np.min([self.sparse.get(
                    'nb_inducing_inputs'), len(self.GP_X)])
            else:
                self.nb_inducing_inputs = int(np.floor(len(self.GP_X) / 10))
                self.sparse.update({'default_nb_inducing_inputs': 'len(X)/10'})
            random_idx = \
                torch.randperm(len(self.GP_X))[:self.nb_inducing_inputs]
            if self.nb_inducing_inputs == 1:
                self.GP_inducing_inputs = reshape_pt1(self.GP_X[random_idx])
            else:
                self.GP_inducing_inputs = reshape_dim1(self.GP_X[random_idx])
            self.variables['GP_inducing_inputs'] = self.GP_inducing_inputs
            if self.sparse.get('method') == 'SVGP':
                self.model = MultitaskSVGPModel(
                    self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                    num_latents=self.num_latents,
                    inducing_points=self.GP_inducing_inputs,
                    output_size=self.GP_Y.shape[1],
                    minibatch_size=self.minibatch_size, **GP_model_options)
            else:
                logging.error('This sparse method is not implemented')
                raise NotImplementedError
        elif self.multioutput_GP:
            logging.warning('MultitaskGaussianLikelihood needed for '
                            'multioutput GPs')
            self.model = MultioutGPModel(
                self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                output_size=self.GP_Y.shape[1])
        elif self.multitask_GP:
            logging.warning('MultitaskGaussianLikelihood needed for '
                            'multitask GPs')
            logging.error('Not implemented yet')
        elif self.sparse:
            if self.sparse.get('nb_inducing_inputs'):
                self.nb_inducing_inputs = np.min([self.sparse.get(
                    'nb_inducing_inputs'), len(self.GP_X)])
            else:
                self.nb_inducing_inputs = int(np.floor(len(self.GP_X) / 10))
                self.sparse.update({'default_nb_inducing_inputs': 'len(X)/10'})
            random_idx = \
                torch.randperm(len(self.GP_X))[:self.nb_inducing_inputs]
            if self.nb_inducing_inputs == 1:
                self.GP_inducing_inputs = reshape_pt1(self.GP_X[random_idx])
            else:
                self.GP_inducing_inputs = reshape_dim1(self.GP_X[random_idx])
            self.variables['GP_inducing_inputs'] = self.GP_inducing_inputs
            if self.sparse.get('method') == 'SparseGP':
                self.model = SparseGPModel(self.GP_X, self.GP_Y,
                                           self.likelihood,
                                           self.kernel, self.GP_inducing_inputs)
            elif self.sparse.get('method') == 'SVGP':
                self.model = SVGPModel(
                    self.GP_X, self.GP_Y, self.likelihood, self.kernel,
                    inducing_points=self.GP_inducing_inputs,
                    minibatch_size=self.minibatch_size, **GP_model_options)
            elif 'kMeans' in self.sparse.get('method'):
                self.model = ExactGPModel(self.GP_X, self.GP_Y, self.likelihood,
                                          self.kernel)
            else:
                logging.error('This sparse method is not implemented')
                raise NotImplementedError
        else:
            self.model = ExactGPModel(self.GP_X, self.GP_Y, self.likelihood,
                                      self.kernel)
        if torch.cuda.is_available() and not self.use_GPy:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        if self.hyperparam_optim and self.step == 1:
            # Fix the starting point for hyperparameter optimization to the
            # parameters at initialization
            if self.hyperparam_optim == 'fixed_start':
                self.start_optim_fixed = copy.deepcopy(self.model.state_dict())

    def update_data(self, new_X=[], new_U=[], new_Y=[]):
        # Update GP with standardized, noisy observation data
        if self.step > 1 and self.memory_saving:
            # TODO update memory saving for torch
            # At each GP update, write only new data to file and read whole
            self.save_intermediate(memory_saving=True)
            X = pd.read_csv(
                os.path.join(self.results_folder, 'X.csv'), sep=',',
                header=None)
            whole_X = torch.as_tensor(X.drop(X.columns[0], axis=1).values)
            U = pd.read_csv(
                os.path.join(self.results_folder, 'U.csv'), sep=',',
                header=None)
            whole_U = torch.as_tensor(U.drop(U.columns[0], axis=1).values)
            Y = pd.read_csv(
                os.path.join(self.results_folder, 'Y.csv'), sep=',',
                header=None)
            whole_Y = torch.as_tensor(Y.drop(Y.columns[0], axis=1).values)
        elif self.step <= 1 and self.memory_saving:
            self.save_intermediate(memory_saving=False)
            whole_X = self.X
            whole_U = self.U
            whole_Y = self.Y
        else:
            whole_X = self.X
            whole_U = self.U
            whole_Y = self.Y

        if (len(new_X) > 0) and (len(new_U) > 0) and (len(new_Y) > 0):
            # Save new data to separate csv
            filename = 'new_X' + '.csv'
            file = pd.DataFrame(new_X.numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            filename = 'new_U' + '.csv'
            file = pd.DataFrame(new_U.numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            filename = 'new_Y' + '.csv'
            file = pd.DataFrame(new_Y.numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            if torch.cuda.is_available() and not self.use_GPy:
                new_X, new_U, new_Y = new_X.cuda(), new_U.cuda(), new_Y.cuda()
            if self.sliding_window_size_each:
                new_X = new_X[-self.sliding_window_size_each:, :]
                new_U = new_U[-self.sliding_window_size_each:, :]
                new_Y = new_Y[-self.sliding_window_size_each:, :]
            if self.sparse and self.sparse.get('method') == 'kMeans_each':
                # Sparsify each subdataset with kMeans clustering, then ExactGP
                # Work on points (x(t), u(t), x(t+dt))
                D = torch.cat((new_X, new_U, new_Y), dim=1)
                k = self.sparse.get('nb_inducing_inputs')
                cl, c = kMeans(D, k, **self.GP_model_options)
                # Filter nans out
                c = c[~torch.any(c.isnan(), dim=1)]
                new_X = c[:, :new_X.shape[1]]
                new_U = c[:, new_X.shape[1]:new_X.shape[1] + new_U.shape[1]]
                new_Y = c[:, new_X.shape[1] + new_U.shape[1]:]
            if self.restart_on_loop:
                # Get rid of continuity between trajs since restart
                self.X = torch.cat((self.X, reshape_pt1(new_X)), dim=0)
                self.U = torch.cat((self.U, reshape_pt1(new_U)), dim=0)
                self.Y = torch.cat((self.Y, reshape_pt1(new_Y)), dim=0)
                whole_X = torch.cat((whole_X, reshape_pt1(new_X)), dim=0)
                whole_U = torch.cat((whole_U, reshape_pt1(new_U)), dim=0)
                whole_Y = torch.cat((whole_Y, reshape_pt1(new_Y)), dim=0)
            else:
                # Get rid of last point of previous traj since no restart
                self.X = torch.cat((self.X[:-1, :], reshape_pt1(new_X)), dim=0)
                self.U = torch.cat((self.U[:-1, :], reshape_pt1(new_U)), dim=0)
                self.Y = torch.cat((self.Y[:-1, :], reshape_pt1(new_Y)), dim=0)
                whole_X = torch.cat((whole_X[:-1, :], reshape_pt1(new_X)),
                                    dim=0)
                whole_U = torch.cat((whole_U[:-1, :], reshape_pt1(new_U)),
                                    dim=0)
                whole_Y = torch.cat((whole_Y[:-1, :], reshape_pt1(new_Y)),
                                    dim=0)
        elif ((len(new_X) > 0) or (len(new_U) > 0) or (
                len(new_Y) > 0)) and not (
                (len(new_X) > 0) or (len(new_U) > 0) or (len(new_Y) > 0)):
            raise ValueError(
                'Only partial new data has been given to re-train the GP. '
                'Please make sure you enter new X, U and Y.')
        self.sample_idx = len(whole_X)  # Nb of samples since start
        pure_X = whole_X.clone()
        pure_U = whole_U.clone()
        pure_Y = whole_Y.clone()

        if self.sliding_window_size:
            whole_X = whole_X[-self.sliding_window_size:, :]
            whole_U = whole_U[-self.sliding_window_size:, :]
            whole_Y = whole_Y[-self.sliding_window_size:, :]

        if self.prior_mean:
            # Only consider residuals = Y - prior_mean(X,U) as output data
            prior_mean_vector = self.prior_mean(whole_X, whole_U,
                                                self.prior_kwargs)
            if self.monitor_experiment:
                for i in range(prior_mean_vector.shape[1]):
                    name = 'GP_prior_mean' + str(i) + '.pdf'
                    plt.plot(prior_mean_vector[:, i].cpu(),
                             label='Prior mean ' + str(i))
                    plt.plot(whole_Y[:, i].cpu(), label='Output data ' + str(i))
                    plt.title('Visualization of prior mean given to GP')
                    plt.legend()
                    plt.xlabel('Time step')
                    plt.ylabel('Prior')
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    if self.verbose:
                        plt.show()
                    plt.close('all')
            whole_Y = reshape_pt1(whole_Y - prior_mean_vector)

        if self.sparse and self.sparse.get('method') == 'kMeans':
            # Sparsify whole dataset with kMeans clustering, then ExactGP
            # Work on points (x(t), u(t), x(t+dt))
            D = torch.cat((whole_X, whole_U, whole_Y), dim=1)
            k = self.sparse.get('nb_inducing_inputs')
            cl, c = kMeans(D, k, **self.GP_model_options)
            # Filter nans out
            c = c[~torch.any(c.isnan(), dim=1)]
            whole_X = c[:, :whole_X.shape[1]]
            whole_U = c[:, whole_X.shape[1]:whole_X.shape[1] + whole_U.shape[1]]
            whole_Y = c[:, whole_X.shape[1] + whole_U.shape[1]:]

        self.scaler_X = StandardScaler(whole_X)
        whole_scaled_X = self.scaler_X.transform(whole_X)
        self.scaler_U = StandardScaler(whole_U)
        whole_scaled_U = self.scaler_U.transform(whole_U)
        if self.no_control:
            # Ignore control for learning
            self.GP_X = whole_scaled_X
        else:
            self.GP_X = torch.cat((whole_scaled_X, whole_scaled_U), dim=1)
        self.scaler_Y = StandardScaler(whole_Y)
        self.GP_Y = self.scaler_Y.transform(whole_Y)

        return pure_X, pure_U, pure_Y

    def update_model(self):
        # Create model (redo at each new learn) and optimize hyperparameters
        self.create_GP_model()
        self.optimize_hyperparameters(self.model)
        if self.use_GPy:
            if self.sparse and not (self.nb_inducing_inputs >= len(self.GP_X)):
                start = len(self.model.param_array) - (len(
                    self.model.kern.param_array) + 1)
                end = len(self.model.param_array)
                if not self.hyperparams.any():
                    self.hyperparams = reshape_pt1(np.array(
                        [self.sample_idx] + [np.copy(self.model.param_array)[i]
                                             for i in range(start, end)]))
                else:
                    self.hyperparams = np.concatenate(
                        (self.hyperparams, reshape_pt1(
                            np.array([self.sample_idx] + [
                                np.copy(self.model.param_array)[i] for i in
                                range(start, end)]))), axis=0)
                if self.monitor_experiment:
                    self.GP_inducing_inputs = reshape_dim1([np.copy(
                        self.model.inducing_inputs)[i] for i in range(len(
                        self.model.inducing_inputs))])
                    df = pd.DataFrame(self.GP_inducing_inputs)
                    df.to_csv(os.path.join(self.results_folder,
                                           'GPy_inducing_inputs.csv'),
                              header=False)
            else:
                if not self.hyperparams.any():
                    self.hyperparams = reshape_pt1(np.array(
                        [self.sample_idx] + [np.copy(self.model.param_array)[i]
                                             for i in range(
                                len(self.model.param_array))]))
                else:
                    self.hyperparams = np.concatenate(
                        (self.hyperparams, reshape_pt1(
                            np.array([self.sample_idx] + [
                                np.copy(self.model.param_array)[i] for i in
                                range(len(self.model.param_array))]))), axis=0)
            return 'GPy update model'

        hyperparam_list = []
        if self.sparse and (self.sparse.get('method') == 'SVGP'):
            named_hyperparams = chain(
                self.model.named_parameters_and_constraints(),
                self.likelihood.named_parameters_and_constraints())
        else:
            named_hyperparams = self.model.named_parameters_and_constraints()
        for name, param, constraint in named_hyperparams:
            if any(k in name for k in ('inducing', 'variational')):
                continue
            value = param if constraint is None else \
                constraint.transform(param)
            value = np.squeeze(value.data.cpu().numpy()).tolist()
            if not isinstance(value, list):
                value = [value]
            if isinstance(value[0], list):
                old_value = value.copy()
                value = []
                for i in range(len(old_value)):
                    value += old_value[i]
            hyperparam_list += value
        if not self.hyperparams.any():
            self.hyperparams = reshape_pt1(np.array(
                [self.sample_idx] + hyperparam_list, dtype=object))
        else:
            self.hyperparams = np.concatenate((
                self.hyperparams, reshape_pt1([self.sample_idx]
                                              + hyperparam_list)), axis=0)

    def optimize_hyperparameters(self, model):
        # Optimize hyperparameters
        if self.use_GPy:
            if self.hyperparam_optim == 'fixed_hyperparameters':
                if self.sparse:
                    # https://github.com/SheffieldML/GPy/issues/623
                    for i in range(len(model.kern.parameters)):
                        model.kern.parameters[i].fix()
                    model.likelihood.variance.fix()
                    model.optimize_restarts(num_restarts=5, verbose=True,
                                            max_iters=100, robust=True)
                else:
                    pass
            elif self.hyperparam_optim == 'fixed_all_parameters':
                pass
            elif self.hyperparam_optim == 'fixed_start':
                model.optimize(start=self.start_optim_fixed,
                               messages=self.verbose,
                               max_iters=100)
            else:
                model.optimize_restarts(num_restarts=5, verbose=True,
                                        max_iters=100, robust=True)
            self.variables['loss'] = torch.tensor([])
            return 'GPy hyperparameter optimization'

        if self.GP_optim_options:
            optim_options = self.GP_optim_options
        else:
            optim_options = {}
        if self.GP_optim_scheduler:
            if self.GP_optim_scheduler_options:
                optim_scheduler_options = self.GP_optim_scheduler_options
            else:
                optim_scheduler_options = {}
            scheduler = self.GP_optim_scheduler
        else:
            scheduler = None
            optim_scheduler_options = {}
        if self.GP_optim_stopper:
            stopper = self.GP_optim_stopper
        else:
            stopper = None
        if self.hyperparam_optim == 'fixed_hyperparameters':
            if self.sparse and not ('kMeans' in self.sparse.get('method')):
                named_hyperparams = model.named_parameters_and_constraints()
                optim_params = set(model.parameters())
                for name, param, constraint in named_hyperparams:
                    if not any(k in name for k in ('inducing', 'variational')):
                        optim_params -= {param}
                optim_params = list(optim_params)
                self.variables['loss'] = torch.tensor(model.optimize(
                    self.likelihood, self.GP_X, self.GP_Y,
                    training_iter=self.GP_optim_training_iter,
                    optimizer=self.GP_optim_method, lr=self.GP_optim_lr,
                    optim_options=optim_options, scheduler=scheduler,
                    scheduler_options=optim_scheduler_options,
                    stopper=stopper, parameters=optim_params,
                    verbose=self.verbose))
            else:
                pass
        elif self.hyperparam_optim == 'fixed_all_parameters':
            pass
        elif self.hyperparam_optim == 'fixed_start':
            model.load_state_dict(self.start_optim_fixed, strict=False)
            self.variables['loss'] = torch.tensor(model.optimize(
                self.likelihood, self.GP_X, self.GP_Y,
                training_iter=self.GP_optim_training_iter,
                optimizer=self.GP_optim_method, lr=self.GP_optim_lr,
                optim_options=optim_options, scheduler=scheduler,
                    scheduler_options=optim_scheduler_options,
                    stopper=stopper, verbose=self.verbose))
        elif self.hyperparam_optim == 'only_likelihood':
            named_hyperparams = model.named_parameters_and_constraints()
            optim_params = set(model.parameters())
            for name, param, constraint in named_hyperparams:
                if not 'likelihood' in name:
                    optim_params -= {param}
            optim_params = list(optim_params)
            self.variables['loss'] = torch.tensor(model.optimize(
                self.likelihood, self.GP_X, self.GP_Y,
                training_iter=self.GP_optim_training_iter,
                optimizer=self.GP_optim_method, lr=self.GP_optim_lr,
                optim_options=optim_options, scheduler=scheduler,
                scheduler_options=optim_scheduler_options, stopper=stopper,
                parameters=optim_params, verbose=self.verbose))
        else:
            self.variables['loss'] = torch.tensor(model.optimize(
                self.likelihood, self.GP_X, self.GP_Y,
                training_iter=self.GP_optim_training_iter,
                optimizer=self.GP_optim_method, lr=self.GP_optim_lr,
                optim_options=optim_options, scheduler=scheduler,
                scheduler_options=optim_scheduler_options, stopper=stopper,
                verbose=self.verbose))
        self.last_optimparams = self.step
        if self.verbose:
            logging.info(model)

    def evaluate_model(self):
        # Record RMSE, average log_likelihood, standardized versions
        if self.memory_saving:
            # Read grid, true prediction and controls from csv
            grid = pd.read_csv(
                os.path.join(self.results_folder, 'Evaluation_grid.csv'),
                sep=',', header=None)
            grid = torch.as_tensor(grid.drop(grid.columns[0], axis=1).values)
            grid_controls = pd.read_csv(
                os.path.join(self.results_folder, 'Grid_controls.csv'),
                sep=',', header=None)
            grid_controls = torch.as_tensor(grid_controls.drop(
                grid_controls.columns[0], axis=1).values)
            true_predicted_grid = pd.read_csv(
                os.path.join(self.results_folder, 'True_predicted_grid.csv'),
                sep=',', header=None)
            true_predicted_grid = torch.as_tensor(true_predicted_grid.drop(
                true_predicted_grid.columns[0], axis=1).values)
            self.grid = grid.reshape(-1, self.X.shape[1])
            self.grid_controls = grid_controls.reshape(-1, self.U.shape[1])
            self.true_predicted_grid = true_predicted_grid.reshape(
                -1, self.Y.shape[1])

        RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
        grid_controls, log_likelihood, stand_log_likelihood = \
            self.compute_l2error_grid(
                grid=self.grid, grid_controls=self.grid_controls,
                true_predicted_grid=self.true_predicted_grid)
        self.RMSE_time = torch.cat((self.RMSE_time, reshape_pt1(
            torch.tensor([torch.tensor(self.sample_idx), RMSE]))), dim=0)
        self.SRMSE_time = torch.cat((self.SRMSE_time, reshape_pt1(
            torch.tensor([torch.tensor(self.sample_idx), SRMSE]))), dim=0)
        self.log_likelihood_time = torch.cat(
            (self.log_likelihood_time, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              log_likelihood]))), dim=0)
        self.stand_log_likelihood_time = torch.cat(
            (self.stand_log_likelihood_time, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              stand_log_likelihood]))), dim=0)

        # Write evaluation results to a file
        # filename = 'Predicted_grid' + str(self.step) + '.csv'
        # file = pd.DataFrame(predicted_grid.numpy())
        # file.to_csv(os.path.join(self.results_folder, filename), header=False)
        if self.step > 1 and self.memory_saving:
            # Delete grid and controls again
            self.grid = self.grid[-1:, :]
            self.grid_controls = self.grid_controls[-1:, :]
            self.true_predicted_grid = self.true_predicted_grid[-1:, :]
        return RMSE_array_dim, RMSE, SRMSE, predicted_grid, \
               true_predicted_grid, grid_controls, log_likelihood, \
               stand_log_likelihood

    def predict(self, x, u, scale=True, only_prior=False):
        # Predict outcome of input, adding prior mean if necessary
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        if scale:
            scaled_x = reshape_pt1(self.scaler_X.transform(x))
            unscaled_x = reshape_pt1(x)
            scaled_u = reshape_pt1(self.scaler_U.transform(u))
            unscaled_u = reshape_pt1(u)
        else:
            scaled_x = reshape_pt1(x)
            unscaled_x = reshape_pt1(self.scaler_X.inverse_transform(x))
            scaled_u = reshape_pt1(u)
            unscaled_u = reshape_pt1(self.scaler_U.inverse_transform(u))
        if only_prior:
            if not self.prior_mean:
                raise ValueError('No prior mean given with only_prior option.')
            # Return prior mean: predicted_x = prior(x)
            prior = reshape_pt1(
                self.prior_mean(unscaled_x, unscaled_u, self.prior_kwargs))
            predicted_mean = prior
            predicted_lowconf = prior
            predicted_uppconf = prior
            predicted_var = torch.tensor([[0.]], device=device)
        else:
            if self.no_control:
                # Ignore control for learning
                GP_x = reshape_pt1(scaled_x)
            else:
                GP_x = torch.cat((reshape_pt1(scaled_x), reshape_pt1(scaled_u)),
                                 dim=1)
            if self.use_GPy:
                mean, var = self.model.predict(GP_x.numpy(),
                                               full_cov=False)
                mean = torch.from_numpy(mean)
                var = torch.ones_like(mean) * torch.from_numpy(var)
            else:
                mean, var = self.model.predict(GP_x, self.likelihood,
                                               full_cov=False)
            # Ignore variance too low
            if torch.max(torch.abs(var)) < 1e-8:
                var = 1e-8 * torch.ones_like(var)
            # No normalization for variance, use uppconf/lowconf instead!!
            predicted_mean = reshape_pt1(
                self.scaler_Y.inverse_transform(mean))
            predicted_var = reshape_pt1(var)
            predicted_lowconf = reshape_pt1(
                self.scaler_Y.inverse_transform(mean - 2 * torch.sqrt(var)))
            predicted_uppconf = reshape_pt1(
                self.scaler_Y.inverse_transform(mean + 2 * torch.sqrt(var)))
            if self.prior_mean:
                # Add prior mean: predicted_x = GP_predicts(x) + prior(x)
                prior = reshape_pt1(
                    self.prior_mean(unscaled_x, unscaled_u, self.prior_kwargs))
                predicted_mean = predicted_mean + prior
                predicted_lowconf = predicted_lowconf + prior
                predicted_uppconf = predicted_uppconf + prior
        return predicted_mean, predicted_var, predicted_lowconf, \
               predicted_uppconf

    def predict_deriv(self, x, u, scale=True, only_x=False, only_prior=False):
        # Predict derivative of posterior distribution d mean / dx, d var /
        # dx at x, adding prior mean if necessary
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        if scale:
            scaled_x = reshape_pt1(self.scaler_X.transform(x))
            unscaled_x = reshape_pt1(x)
            scaled_u = reshape_pt1(self.scaler_U.transform(u))
            unscaled_u = reshape_pt1(u)
        else:
            scaled_x = reshape_pt1(x)
            unscaled_x = reshape_pt1(self.scaler_X.inverse_transform(x))
            scaled_u = reshape_pt1(u)
            unscaled_u = reshape_pt1(self.scaler_U.inverse_transform(u))
        if only_prior:
            # Return prior mean: predicted_x = prior(x)
            prior = reshape_pt1(
                self.prior_mean(unscaled_x, unscaled_u, self.prior_kwargs))
            predicted_mean = prior
            predicted_lowconf = prior
            predicted_uppconf = prior
            predicted_var = torch.tensor([[0.]], device=device)
        else:
            if self.no_control:
                # Ignore control for learning
                GP_x = reshape_pt1(scaled_x)
            else:
                GP_x = torch.cat((reshape_pt1(scaled_x), reshape_pt1(scaled_u)),
                                 dim=1)
            if self.use_GPy:
                mean0, var0 = self.model.predict(GP_x.numpy(),
                                               full_cov=False)
                mean0 = torch.from_numpy(mean0)
                var0 = torch.ones_like(mean0) * torch.from_numpy(var0)
                mean, var = self.model.predictive_gradients(GP_x.numpy(),
                                                            self.kernel)
                mean = torch.from_numpy(mean)
                # var = torch.ones_like(mean) * torch.from_numpy(var)
                var = torch.mul(torch.ones_like(mean),
                                torch.unsqueeze(torch.from_numpy(var), dim=2))
                mean, var = torch.transpose(mean, 1, 2), \
                            torch.transpose(var, 1, 2)  # (N, dy, dx)
            else:
                mean0, var0 = self.model.predict(GP_x, self.likelihood,
                                                 full_cov=False)
                mean, var = self.model.predict_gradient(GP_x, self.likelihood)
            # Ignore variance too low
            if torch.max(torch.abs(var)) < 1e-8:
                var = 1e-8 * torch.ones_like(var)
            # Denormalize and reshape Jacobian to obtain (N, dy, dx)
            # No normalization for variance, use uppconf/lowconf instead!!
            if self.no_control:
                sigmaX = self.scaler_X._scale
            else:
                sigmaX = torch.cat((self.scaler_X._scale,
                                    self.scaler_U._scale))
            sigmaY = self.scaler_Y._scale
            if len(mean.shape) == 2:
                # dy = 1: mean and var shape (N, dx)
                mean = reshape_pt1(mean)
                predicted_mean = sigmaY * mean / sigmaX
                predicted_var = reshape_pt1(var)
                sigmainv = 1 / torch.sqrt(var0)  # (N, 1)
                predicted_lowconf = sigmaY * (
                        mean - sigmainv * predicted_var) / sigmaX
                predicted_uppconf = sigmaY * (
                        mean + sigmainv * predicted_var) / sigmaX
            else:
                if mean.shape[0] == 1:
                    # N = 1: mean and var shape (dy, dx)
                    mean = torch.squeeze(mean, 0)
                    predicted_mean = reshape_pt1(
                        torch.diag(sigmaY) @ mean @ torch.diag(1 / sigmaX))
                    predicted_var = torch.squeeze(var, 0)
                    sigmainv = 1 / torch.sqrt(var0).t()  # (dy, 1)
                    predicted_lowconf = reshape_pt1(
                        torch.diag(sigmaY) @ (mean - sigmainv * predicted_var)
                        @ torch.diag(1 / sigmaX))
                    predicted_uppconf = reshape_pt1(
                        torch.diag(sigmaY) @ (mean + sigmainv * predicted_var)
                        @ torch.diag(1 / sigmaX))
                else:
                    predicted_mean = torch.diag(sigmaY) @ mean @ \
                                     torch.diag(1 / sigmaX)
                    predicted_var = reshape_pt1(var)
                    sigmainv = 1 / torch.sqrt(var0)  # (N, dy)
                    predicted_lowconf = torch.diag(sigmaY) @ (
                            mean - sigmainv * predicted_var) @ torch.diag(
                        1 / sigmaX)
                    predicted_uppconf = torch.diag(sigmaY) @ (
                            mean + sigmainv * predicted_var) @ torch.diag(
                        1 / sigmaX)
            if self.prior_mean and not self.prior_mean_deriv:
                logging.error('A prior function was given and used in the '
                              'mean, but its derivative was not given for '
                              'the derivative predictions.')
            if self.prior_mean_deriv:
                # Add prior mean: predicted_x = GP_predicts(x) + prior(x)
                prior = reshape_pt1(self.prior_mean_deriv(
                    unscaled_x, unscaled_u, self.prior_kwargs))
                predicted_mean = predicted_mean + prior
                predicted_lowconf = predicted_lowconf + prior
                predicted_uppconf = predicted_uppconf + prior
        if only_x:
            return predicted_mean[:, :x.shape[1]], \
                   predicted_var[:, :x.shape[1]], \
                   predicted_lowconf[:, :x.shape[1]], \
                   predicted_uppconf[:, :x.shape[1]]
        else:
            return predicted_mean, predicted_var, predicted_lowconf, \
                   predicted_uppconf

    def predict_euler_Michelangelo(self, x, u, scale=True, only_prior=False):
        # Predict x_t+1 from prediction of y_t in Michelangelo framework
        # TODO better than Euler?
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = torch.diag_embed(torch.ones(x.shape[1] - 1, device=device),
                             offset=1)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = 1
        ABmult = torch.matmul(x, A.t()) + torch.matmul(predicted_mean, B.t())
        ABmult_lowconf = torch.matmul(x, A.t()) + torch.matmul(
            predicted_lowconf, B.t())
        ABmult_uppconf = torch.matmul(x, A.t()) + torch.matmul(
            predicted_uppconf, B.t())
        predicted_mean_euler = reshape_pt1(x) + self.dt * ABmult  # + u)
        predicted_mean_euler_lowconf = reshape_pt1(x) + self.dt * \
                                       ABmult_lowconf  # + u)
        predicted_mean_euler_uppconf = reshape_pt1(x) + self.dt * \
                                       ABmult_uppconf  # + u)
        return predicted_mean_euler, predicted_var, \
               predicted_mean_euler_lowconf, predicted_mean_euler_uppconf

    def true_dynamics_euler_Michelangelo(self, x, u):
        # Compute x_t+1 from true y_t in Michelangelo framework
        # TODO better than Euler?
        device = x.device
        true_mean = reshape_pt1(self.true_dynamics(x, u))
        A = torch.diag_embed(torch.ones(x.shape[1] - 1, device=device),
                             offset=1)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = 1
        ABmult = torch.matmul(x, A.t()) + torch.matmul(true_mean, B.t())
        true_mean_euler = reshape_pt1(x) + self.dt * ABmult  # + u)
        return true_mean_euler

    def predict_euler_discrete_justvelocity(self, x, u, scale=True,
                                            only_prior=False):
        # Predict x_t+1 from prediction of velocity_t with chain of integrators
        # TODO better than Euler?
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = torch.eye(x.shape[1], device=device) + self.dt * torch.diag_embed(
            torch.ones(x.shape[1] - 1, device=device), offset=1)
        A[-1] = torch.zeros((1, x.shape[1]), device=device)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = 1
        predicted_mean_euler = torch.matmul(x, A.t()) + \
                               torch.matmul(predicted_mean, B.t())
        predicted_mean_euler_lowconf = torch.matmul(x, A.t()) + \
                                       torch.matmul(predicted_lowconf, B.t())
        predicted_mean_euler_uppconf = torch.matmul(x, A.t()) + \
                                       torch.matmul(predicted_uppconf, B.t())
        return reshape_pt1(predicted_mean_euler), reshape_pt1(predicted_var), \
               reshape_pt1(predicted_mean_euler_lowconf), \
               reshape_pt1(predicted_mean_euler_uppconf)

    def true_dynamics_euler_discrete_justvelocity(self, x, u):
        # Compute x_t+1 from true velocity_t with chain of integrators
        # TODO better than Euler?
        device = x.device
        true_mean = reshape_pt1(self.true_dynamics(x, u))
        A = torch.eye(x.shape[1], device=device) + self.dt * torch.diag_embed(
            torch.ones(x.shape[1] - 1, device=device), offset=1)
        A[-1] = torch.zeros((1, x.shape[1]), device=device)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = 1
        true_mean_euler = torch.matmul(x, A.t()) + torch.matmul(
            true_mean, B.t())
        return reshape_pt1(true_mean_euler)

    def predict_euler_continuous_justvelocity(self, x, u, scale=True,
                                              only_prior=False):
        # Predict x_t+1 from prediction of xdot_t with chain of integrators
        # TODO better than Euler?
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = torch.eye(x.shape[1], device=device) + self.dt * torch.diag_embed(
            torch.ones(x.shape[1] - 1, device=device), offset=1)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = self.dt
        predicted_mean_euler = torch.matmul(x, A.t()) + \
                               torch.matmul(predicted_mean, B.t())
        predicted_mean_euler_lowconf = torch.matmul(x, A.t()) + \
                                       torch.matmul(predicted_lowconf, B.t())
        predicted_mean_euler_uppconf = torch.matmul(x, A.t()) + \
                                       torch.matmul(predicted_uppconf, B.t())
        return reshape_pt1(predicted_mean_euler), reshape_pt1(predicted_var), \
               reshape_pt1(predicted_mean_euler_lowconf), \
               reshape_pt1(predicted_mean_euler_uppconf)

    def true_dynamics_euler_continuous_justvelocity(self, x, u):
        # Compute x_t+1 from true xdot_t with chain of integrators
        # TODO better than Euler?
        device = x.device
        true_mean = reshape_pt1(self.true_dynamics(x, u))
        A = torch.eye(x.shape[1], device=device) + self.dt * torch.diag_embed(
            torch.ones(x.shape[1] - 1, device=device), offset=1)
        B = torch.zeros((x.shape[1], 1), device=device)
        B[-1] = self.dt
        true_mean_euler = torch.matmul(x, A.t()) + \
                          torch.matmul(true_mean, B.t())
        return reshape_pt1(true_mean_euler)

    def compute_l2error_grid(self, grid, grid_controls, true_predicted_grid,
                             use_euler=None):
        # RMSE, at fixed control, between real and GP predictions
        # Average log probability of real prediction coming from GP predicted
        # distribution (in scaled domain)
        if not use_euler:
            predicted_mean, predicted_var, predicted_lowconf, \
            predicted_uppconf = self.predict(grid, grid_controls)
        elif use_euler == 'Michelangelo':
            predicted_mean, predicted_var, predicted_lowconf, \
            predicted_uppconf = \
                self.predict_euler_Michelangelo(grid, grid_controls)
        elif use_euler == 'discrete_justvelocity':
            predicted_mean, predicted_var, predicted_lowconf, \
            predicted_uppconf = \
                self.predict_euler_discrete_justvelocity(grid, grid_controls)
        elif use_euler == 'continuous_justvelocity':
            predicted_mean, predicted_var, predicted_lowconf, \
            predicted_uppconf = \
                self.predict_euler_continuous_justvelocity(grid, grid_controls)
        else:
            logging.error('This version of Euler/discretized prediction '
                          'is not implemented.')
        l2_error_array = torch.square(true_predicted_grid - predicted_mean)
        predicted_grid = torch.cat((predicted_mean, predicted_var), dim=1)

        self.variables['l2_error_array'] = l2_error_array
        RMSE_array_dim = torch.sqrt(torch.mean(l2_error_array, dim=0))
        RMSE = RMS(true_predicted_grid - predicted_mean)
        var = torch.var(true_predicted_grid)  # would be better det(covar)
        SRMSE = RMSE / var
        log_likelihood = log_multivariate_normal_likelihood(
            true_predicted_grid, predicted_mean, predicted_var)
        if reshape_pt1(self.scaler_Y._mean).shape[1] == \
                true_predicted_grid.shape[1]:
            mean_vector = reshape_pt1(
                torch.repeat_interleave(reshape_pt1(self.scaler_Y._mean),
                                        len(true_predicted_grid), dim=0))
            var_vector = reshape_pt1(torch.repeat_interleave(reshape_pt1(
                self.scaler_Y._var), len(true_predicted_grid), dim=0))
        else:
            mean_vector = reshape_pt1(
                torch.repeat_interleave(reshape_pt1(self.scaler_X._mean),
                                        len(true_predicted_grid), dim=0))
            var_vector = reshape_pt1(torch.repeat_interleave(reshape_pt1(
                self.scaler_X._var), len(true_predicted_grid), dim=0))
        stand_log_likelihood = log_likelihood - \
                               log_multivariate_normal_likelihood(
                                   true_predicted_grid, mean_vector,
                                   var_vector)
        return RMSE_array_dim, RMSE.cpu(), SRMSE.cpu(), predicted_grid, \
               true_predicted_grid, grid_controls, log_likelihood.cpu(), \
               stand_log_likelihood.cpu()

    def create_grid(self, init_state, init_control, constrain_u, grid_inf,
                    grid_sup):
        # Create random grid for evaluation
        # https://stackoverflow.com/questions/45583274/how-to-generate-an-n-dimensional-grid-in-python
        dx = init_state.shape[1]
        du = init_control.shape[1]
        if constrain_u:
            umin = np.min(constrain_u)
            umax = np.max(constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = torch.min(self.U, dim=0).values
            umax = torch.max(self.U, dim=0).values
        if not self.ground_truth_approx:
            nb_points = int(np.ceil(np.max([10 ** 4 / dx, 500])))
            grid = reshape_pt1(torch.rand((nb_points, dx), device=self.device)
                               * (grid_sup - grid_inf) + grid_inf)
            grid_controls = reshape_pt1(torch.rand(
                (nb_points, du), device=self.device) * (umax - umin) + umin)
        else:
            nb_points = int(np.ceil(np.min([len(self.X), 1000])))
            self.grid_random_idx = torch.randperm(len(self.X))[:nb_points]
            grid = reshape_pt1(self.X[self.grid_random_idx])
            grid_controls = reshape_pt1(self.U[self.grid_random_idx])
        return grid, grid_controls

    def create_true_predicted_grid(self, grid, grid_controls):
        true_predicted_grid = torch.zeros((len(grid), self.Y.shape[1]),
                                          device=self.device)
        if not self.ground_truth_approx:
            for idx, x in enumerate(true_predicted_grid):
                control = reshape_pt1(grid_controls[idx])
                x = reshape_pt1(grid[idx])
                true_predicted_grid[idx] = reshape_pt1(
                    self.true_dynamics(x, control))
        else:
            true_predicted_grid = reshape_pt1(self.Y[self.grid_random_idx])
        return true_predicted_grid

    def read_grid_variables(self, results_folder):
        for key, val in self.grid_variables.items():
            filename = str(key) + '.csv'
            data = pd.read_csv(
                os.path.join(results_folder, filename), sep=',', header=None)
            self.grid_variables[str(key)] = torch.as_tensor(data.drop(
                data.columns[0], axis=1).values, device=self.device)

    def save_grid_variables(self, grid, grid_controls, true_predicted_grid,
                            results_folder):
        if torch.is_tensor(grid):
            grid = pd.DataFrame(grid.cpu().numpy())
            grid.to_csv(os.path.join(results_folder, 'Evaluation_grid.csv'),
                        header=False)
            grid_controls = pd.DataFrame(grid_controls.cpu().numpy())
            grid_controls.to_csv(os.path.join(
                results_folder, 'Grid_controls.csv'), header=False)
            true_predicted_grid = pd.DataFrame(
                true_predicted_grid.cpu().numpy())
            true_predicted_grid.to_csv(os.path.join(
                results_folder, 'True_predicted_grid.csv'), header=False)
        else:
            grid = pd.DataFrame(grid)
            grid.to_csv(os.path.join(results_folder, 'Evaluation_grid.csv'),
                        header=False)
            grid_controls = pd.DataFrame(grid_controls)
            grid_controls.to_csv(os.path.join(
                results_folder, 'Grid_controls.csv'), header=False)
            true_predicted_grid = pd.DataFrame(true_predicted_grid)
            true_predicted_grid.to_csv(os.path.join(
                results_folder, 'True_predicted_grid.csv'), header=False)
        if self.memory_saving:
            self.grid = self.grid[-1:, :]
            self.grid_controls = self.grid_controls[-1:, :]
            self.true_predicted_grid = self.true_predicted_grid[-1:, :]

    def create_rollout_list(self):
        # # folder = '../Figures' \
        # #          '/Continuous/Duffing/Discrete_model' \
        # #          '/GP_justvelocity_highgain_observer_noisy_inputs/10_pass' \
        # #          '/10_rollouts/Good6_manuscript_500samples_t0-30_1e-05meas_0process_uclip0.4/Loop_0'
        # folder = '../Figures/Continuous/Mass_spring_mass/Discrete_model' \
        #          '/GP_justvelocity_highgain_observer_noisy_inputs/10_pass' \
        #          '/10_rollouts/Good6_manuscript_250samples_t0-15_1e-08meas_1e-08process_uclip0.4/Loop_0'
        # return self.read_rollout_list(
        #     results_folder=folder, nb_rollouts=self.nb_rollouts,
        #     step=self.step, folder_title=None,
        #     save=self.results_folder)  # TODO change back!
        rollout_list = []
        if self.constrain_u:
            umin = np.min(self.constrain_u)
            umax = np.max(self.constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = torch.min(self.U, dim=0).values
            umax = torch.max(self.U, dim=0).values
        # Quite slow, parallelize a bit?
        for controller, current_nb_rollouts in self.rollout_controller.items():
            i = 0
            while i < current_nb_rollouts:
                if not self.ground_truth_approx:
                    time_vector = torch.arange(0., self.rollout_length) * \
                                  self.dt
                    init_state = reshape_pt1(
                        torch.rand((1, self.X.shape[1])) * (
                                self.grid_sup - self.grid_inf) + self.grid_inf)
                    true_mean = torch.zeros((self.rollout_length + 1,
                                             init_state.shape[1]))
                    true_mean[0] = init_state.clone()
                    # Define control_traj depending on current controller
                    if controller == 'random':
                        control_traj = reshape_dim1(torch.rand((
                            self.rollout_length, self.U.shape[1])) * (
                                                            umax - umin) + umin)
                    elif controller == 'sin_controller_1D':
                        control_traj = sin_controller_1D(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    elif controller == 'sin_controller_02D':
                        control_traj = sin_controller_02D(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    elif controller == 'null_controller':
                        control_traj = null_controller(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    else:
                        raise ValueError(
                            'Controller for rollout is not defined. Available '
                            'options are random, sin_controller_1D, '
                            'sin_controller_02D, null_controller.')
                    for t in range(self.rollout_length):
                        # True and predicted trajectory over time
                        control = control_traj[t]
                        if 'Michelangelo' in self.system:
                            xnext_true = self.true_dynamics_euler_Michelangelo(
                                reshape_pt1(
                                    true_mean[t]), reshape_pt1(control))
                        elif ('justvelocity' in self.system) and not \
                                self.continuous_model:
                            xnext_true = \
                                self.true_dynamics_euler_discrete_justvelocity(
                                    reshape_pt1(true_mean[t]),
                                    reshape_pt1(control))
                        elif 'justvelocity' in self.system and \
                                self.continuous_model:
                            xnext_true = \
                                self.true_dynamics_euler_continuous_justvelocity(
                                    reshape_pt1(true_mean[t]),
                                    reshape_pt1(control))
                        else:
                            xnext_true = reshape_pt1(self.true_dynamics(
                                reshape_pt1(true_mean[t]),
                                reshape_pt1(control)))
                        true_mean[t + 1] = xnext_true
                    max = torch.max(torch.abs(true_mean))
                    if max.numpy() > self.max_rollout_value:
                        # If true trajectory diverges, ignore this rollout
                        logging.warning(
                            'Ignored a rollout with diverging true '
                            'trajectory, with initial state ' + str(
                                init_state) + ' and maximum reached absolute '
                                              'value ' + str(max))
                        continue
                    i += 1
                    rollout_list.append([init_state, control_traj, true_mean])
                else:
                    if self.step > 0:
                        # Only initialize rollout list at beginning of each fold
                        return self.rollout_list
                    # If no ground truth, rollouts are subsets of train data
                    if i == 0:
                        # Initial rollout same as data
                        init_state = reshape_pt1(self.init_state)
                        true_mean = reshape_pt1(
                            self.X[:self.rollout_length + 1, :])
                        control_traj = reshape_pt1(
                            self.U[:self.rollout_length, :])
                    else:
                        # Next rollouts start, control random subset of data
                        start_idx = torch.randint(0, len(self.U) -
                                                  self.rollout_length,
                                                  size=(1,))
                        init_state = reshape_pt1(self.X[start_idx])
                        true_mean = reshape_pt1(
                            self.X[start_idx:start_idx +
                                             self.rollout_length + 1, :])
                        control_traj = reshape_pt1(
                            self.U[start_idx:start_idx +
                                             self.rollout_length, :])
                    rollout_list.append([init_state, control_traj, true_mean])
                    i += 1
        return rollout_list

    def read_rollout_list(self, results_folder, nb_rollouts, step,
                          folder_title=None, save=None):
        if not folder_title:
            folder = os.path.join(results_folder, 'Rollouts_' + str(step))
        else:
            folder = os.path.join(results_folder, folder_title + '_' +
                                  str(step))
        rollout_list = []
        for i in range(nb_rollouts):
            rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
            filename = 'Init_state.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            init_state = data.drop(data.columns[0], axis=1).values
            filename = 'Control_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            control_traj = data.drop(data.columns[0], axis=1).values
            filename = 'True_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            true_mean = data.drop(data.columns[0], axis=1).values
            if save:
                # Save these variables to another folder
                if not folder_title:
                    save_folder = os.path.join(save, 'Rollouts_' + str(step))
                else:
                    save_folder = os.path.join(
                        save, folder_title + '_' + str(step))
                save_rollout_folder = os.path.join(
                    save_folder, 'Rollout_' + str(i))
                os.makedirs(save_rollout_folder, exist_ok=True)
                filename = 'Init_state.csv'
                file = pd.DataFrame(init_state)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
                filename = 'Control_traj.csv'
                file = pd.DataFrame(control_traj)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
                filename = 'True_traj.csv'
                file = pd.DataFrame(true_mean)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
            rollout_list.append([
                torch.as_tensor(init_state, device=self.device),
                torch.as_tensor(control_traj, device=self.device),
                torch.as_tensor(true_mean, device=self.device)])
        return rollout_list

    def evaluate_rollouts(self, only_prior=False):
        if len(self.rollout_list) == 0:
            return 0
        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         only_prior=only_prior)
        self.specs['nb_rollouts'] = self.nb_rollouts
        self.specs['rollout_length'] = self.rollout_length
        self.specs['rollout_RMSE'] = rollout_RMSE
        self.specs['rollout_SRMSE'] = rollout_SRMSE
        self.specs['rollout_log_AL'] = rollout_log_AL
        self.specs['rollout_stand_log_AL'] = rollout_stand_log_AL
        self.rollout_RMSE = \
            torch.cat((self.rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.rollout_SRMSE = \
            torch.cat((self.rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.rollout_log_AL = \
            torch.cat((self.rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.rollout_stand_log_AL = \
            torch.cat((self.rollout_stand_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_stand_log_AL]))), dim=0)
        self.variables['rollout_RMSE'] = self.rollout_RMSE
        self.variables['rollout_SRMSE'] = self.rollout_SRMSE
        self.variables['rollout_log_AL'] = self.rollout_log_AL
        self.variables['rollout_stand_log_AL'] = self.rollout_stand_log_AL
        plot_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = concatenate_lists(self.rollout_list,
                                                      rollout_list)
            save_rollout_variables(
                self, self.results_folder, self.nb_rollouts,
                complete_rollout_list, step=self.step - 1, results=True,
                ground_truth_approx=self.ground_truth_approx,
                plots=self.monitor_experiment)

    def evaluate_kalman_rollouts(self, observer, observe_data,
                                 discrete_observer, no_GP_in_observer=False,
                                 only_prior=False):
        if len(self.rollout_list) == 0:
            return 0

        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory, but in closed loop
        # by correcting with observations at each time step
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         observer=observer, observe_data=observe_data,
                         discrete_observer=discrete_observer, kalman=True,
                         no_GP_in_observer=no_GP_in_observer,
                         only_prior=only_prior)
        self.specs['kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['kalman_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.kalman_rollout_RMSE = \
            torch.cat((self.kalman_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.kalman_rollout_SRMSE = \
            torch.cat((self.kalman_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_SRMSE]))),
                      dim=0)
        self.kalman_rollout_log_AL = \
            torch.cat((self.kalman_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_log_AL]))),
                      dim=0)
        self.kalman_rollout_stand_log_AL = \
            torch.cat((self.kalman_rollout_stand_log_AL, reshape_pt1(
                torch.tensor(
                    [torch.tensor(self.sample_idx), rollout_stand_log_AL]))),
                      dim=0)
        self.variables['kalman_rollout_RMSE'] = self.kalman_rollout_RMSE
        self.variables['kalman_rollout_SRMSE'] = \
            self.kalman_rollout_SRMSE
        self.variables['kalman_rollout_log_AL'] = \
            self.kalman_rollout_log_AL
        self.variables['kalman_rollout_stand_log_AL'] = \
            self.kalman_rollout_stand_log_AL
        plot_kalman_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = concatenate_lists(self.rollout_list,
                                                      rollout_list)
            save_kalman_rollout_variables(
                self, self.results_folder, self.nb_rollouts,
                complete_rollout_list, step=self.step - 1,
                ground_truth_approx=self.ground_truth_approx,
                plots=self.monitor_experiment)

    def evaluate_closedloop_rollouts(self, observer, observe_data,
                                     no_GP_in_observer=False):
        if len(self.rollout_list) == 0:
            return 0

        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory, but in closed loop
        # by correcting with observations at each time step
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         observer=observer, observe_data=observe_data,
                         closedloop=True, no_GP_in_observer=no_GP_in_observer)
        self.specs['closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['closedloop_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.closedloop_rollout_RMSE = \
            torch.cat((self.closedloop_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.closedloop_rollout_SRMSE = \
            torch.cat((self.closedloop_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_SRMSE]))),
                      dim=0)
        self.closedloop_rollout_log_AL = \
            torch.cat((self.closedloop_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_log_AL]))),
                      dim=0)
        self.closedloop_rollout_stand_log_AL = torch.cat((
            self.closedloop_rollout_stand_log_AL, reshape_pt1(
                torch.tensor(
                    [torch.tensor(self.sample_idx), rollout_stand_log_AL]))),
            dim=0)
        self.variables['closedloop_rollout_RMSE'] = \
            self.closedloop_rollout_RMSE
        self.variables['closedloop_rollout_SRMSE'] = \
            self.closedloop_rollout_SRMSE
        self.variables['closedloop_rollout_log_AL'] = \
            self.closedloop_rollout_log_AL
        self.variables['closedloop_rollout_stand_log_AL'] = \
            self.closedloop_rollout_stand_log_AL
        plot_closedloop_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = concatenate_lists(self.rollout_list,
                                                      rollout_list)
            save_closedloop_rollout_variables(
                self, self.results_folder, self.nb_rollouts,
                complete_rollout_list, step=self.step - 1,
                ground_truth_approx=self.ground_truth_approx,
                plots=self.monitor_experiment)

    def read_variables(self, results_folder):
        for key, val in self.variables.items():
            if key.startswith('test_') or key.startswith('val_') or (
                    'rollout' in key):
                # Avoid saving all test and validation variables, and only
                # save rollout variables in rollout functions
                continue
            filename = str(key) + '.csv'
            data = pd.read_csv(
                os.path.join(results_folder, filename), sep=',', header=None)
            self.variables[str(key)] = torch.as_tensor(data.drop(
                data.columns[0], axis=1).values)

    def read_anim_variables(self, results_folder):
        for key, val in self.variables.items():
            if (key == 'X') or (key == 'Y'):
                filename = str(key) + '.csv'
                data = pd.read_csv(
                    os.path.join(results_folder, filename), sep=',',
                    header=None)
                self.variables[str(key)] = torch.as_tensor(data.drop(
                    data.columns[0], axis=1).values)

    def read_control(self, results_folder):
        for key, val in self.variables.items():
            if (key == 'U'):
                filename = str(key) + '.csv'
                data = pd.read_csv(
                    os.path.join(results_folder, filename), sep=',',
                    header=None)
                self.variables[str(key)] = torch.as_tensor(data.drop(
                    data.columns[0], axis=1).values)

    def cut_anim_variables(self):
        for key, val in self.variables.items():
            if (key == 'X') or (key == 'Y'):
                self.variables[key] = self.variables[key][-1:, :]

    def set_results_folder(self, folder):
        # Change results folder and copy grid variables and log there
        self.read_grid_variables(self.results_folder)
        self.save_grid_variables(self.grid, self.grid_controls,
                                 self.true_predicted_grid, folder)
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        save_rollout_variables(self, folder, self.nb_rollouts,
                               self.rollout_list, step=self.step,
                               ground_truth_approx=self.ground_truth_approx,
                               plots=self.monitor_experiment)
        save_log(folder)
        self.results_folder = folder

    def set_dyn_kwargs(self, new_dyn_kwargs):
        # Change dyn kwargs
        self.dyn_kwargs = new_dyn_kwargs
        self.specs['dyn_kwargs'] = new_dyn_kwargs
        self.specs.update(new_dyn_kwargs)

    def set_config(self, new_config: Config):
        # Change config
        self.config = new_config
        self.specs.update(new_config)

    def save_intermediate(self, memory_saving=False, ignore_first=False):
        # Retrieve and save all intermediate variables
        self.variables['X'] = self.X
        self.variables['Y'] = self.Y
        self.variables['U'] = self.U
        self.variables['Computation_time'] = self.time
        self.specs['sparse'] = self.sparse
        if self.monitor_experiment:
            self.variables['RMSE_time'] = self.RMSE_time
            self.variables['SRMSE_time'] = self.SRMSE_time
            self.variables['log_AL_time'] = self.log_likelihood_time
            self.variables['stand_log_AL_time'] = self.stand_log_likelihood_time

        # Store results and parameters in files
        if self.step > 1:
            specs_file = os.path.join(self.results_folder, 'Specifications.txt')
            with open(specs_file, 'w') as f:
                for key, val in self.specs.items():
                    if key == 'kernel':
                        if self.use_GPy:
                            for i in range(len(self.model.kern.parameters)):
                                print(self.model.kern.parameters[i], file=f)
                        else:
                            print_parameters(self.model, file=f)
                    else:
                        print(key, ': ', val, file=f)
        for key, val in self.variables.items():
            if key.startswith('test_') or key.startswith('val_') or (
                    'rollout' in key):
                # Avoid saving all test and validation variables, and only
                # save rollout variables in rollout functions
                continue
            filename = str(key) + '.csv'
            if memory_saving:
                # Append only new values to csv
                if self.step > 1:
                    if ignore_first:
                        file = pd.DataFrame(val[1:, :].numpy())
                    else:
                        file = pd.DataFrame(val.numpy())
                else:
                    file = pd.DataFrame(val.numpy())
                file.to_csv(os.path.join(self.results_folder, filename),
                            mode='a', header=False)
            else:
                file = pd.DataFrame(val.numpy())
                file.to_csv(os.path.join(self.results_folder, filename),
                            header=False)

        if memory_saving:
            # Keep only last value in variable
            for key, val in self.variables.items():
                self.variables[key] = self.variables[key][-1:, :]
            for key, val in self.grid_variables.items():
                self.grid_variables[key] = self.grid_variables[key][-1:, :]
            self.X = self.X[-1:, :]
            self.Y = self.Y[-1:, :]
            self.U = self.U[-1:, :]
            self.time = self.time[-1:, :]
            if self.monitor_experiment:
                self.RMSE_time = self.RMSE_time[-1:, :]
                self.SRMSE_time = self.SRMSE_time[-1:, :]
                self.log_likelihood_time = self.log_likelihood_time[-1:, :]
                self.stand_log_likelihood_time = self.stand_log_likelihood_time[
                                                 -1:, :]

    def save_folder(self, results_folder):
        # Save all variables in a folder, plot results over time
        self.variables['X'] = self.X
        self.variables['Y'] = self.Y
        self.variables['U'] = self.U
        self.variables['Computation_time'] = self.time
        self.specs['model'] = self.model
        self.specs['X_mean'] = self.scaler_X._mean
        self.specs['X_var'] = self.scaler_X._var
        self.specs['U_mean'] = self.scaler_U._mean
        self.specs['U_var'] = self.scaler_U._var
        self.specs['Y_mean'] = self.scaler_Y._mean
        self.specs['Y_var'] = self.scaler_Y._var
        self.specs['sparse'] = self.sparse

        # Store results and parameters in files
        specs_file = os.path.join(results_folder, 'Specifications.txt')
        with open(specs_file, 'w') as f:
            for key, val in self.specs.items():
                if key == 'kernel':
                    if self.use_GPy:
                        for i in range(len(self.model.kern.parameters)):
                            print(self.model.kern.parameters[i], file=f)
                    else:
                        print_parameters(self.model, file=f)
                else:
                    print(key, ': ', val, file=f)
        if self.monitor_experiment:
            for key, val in self.variables.items():
                if key.startswith('test_') or key.startswith('val_') or (
                        'rollout' in key):
                    # Avoid saving all test and validation variables, and only
                    # save rollout variables in rollout functions
                    continue
                filename = str(key) + '.csv'
                file = pd.DataFrame(val.cpu().numpy())
                file.to_csv(os.path.join(results_folder, filename),
                            header=False)
            filename = 'Hyperparameters.csv'
            file = pd.DataFrame(self.hyperparams)
            file.to_csv(os.path.join(results_folder, filename), header=False)
        else:
            for key, val in self.variables.items():
                if any(key == k for k in ('X', 'U', 'Y')):
                    filename = str(key) + '.csv'
                    file = pd.DataFrame(val.cpu().numpy())
                    file.to_csv(os.path.join(results_folder, filename),
                                header=False)

    def save(self):
        # Evaluate model
        if self.monitor_experiment:
            l2_error, RMSE, SRMSE, self.grid_variables['Predicted_grid'], \
            self.grid_variables['True_predicted_grid'], _, log_likelihood, \
            stand_log_likelihood = self.evaluate_model()
            self.specs['l2_error'] = l2_error
            self.specs['RMSE'] = RMSE
            self.specs['SRMSE'] = SRMSE
            self.specs['log_AL'] = log_likelihood
            self.specs['stand_log_AL'] = stand_log_likelihood

            filename = 'Predicted_grid.csv'
            file = pd.DataFrame(
                self.grid_variables['Predicted_grid'].cpu().numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
        # Run rollouts
        self.evaluate_rollouts()

        # Update all evaluation variables
        self.variables['RMSE_time'] = self.RMSE_time
        self.variables['SRMSE_time'] = self.SRMSE_time
        self.variables['log_likelihood_time'] = self.log_likelihood_time
        self.variables[
            'stand_log_likelihood_time'] = self.stand_log_likelihood_time
        self.variables['RMSE_time'] = self.RMSE_time
        self.variables['SRMSE_time'] = self.SRMSE_time
        self.variables['log_AL_time'] = self.log_likelihood_time
        self.variables['stand_log_AL_time'] = self.stand_log_likelihood_time

        # Save updated evaluation variables along with all other variables
        self.save_folder(self.results_folder)

        # If memory saving, read complete variables
        if self.step > 1 and self.memory_saving:
            self.save_intermediate(memory_saving=True, ignore_first=True)
            self.read_grid_variables(self.results_folder)
            self.read_variables(self.results_folder)
            self.X = self.variables['X']
            self.Y = self.variables['Y']
            self.U = self.variables['U']
            self.time = self.variables['Computation_time']
            self.RMSE_time = self.variables['RMSE_time']
            self.SRMSE_time = self.variables['SRMSE_time']
            self.log_likelihood_time = self.variables['log_AL_time']
            self.stand_log_likelihood_time = self.variables[
                'stand_log_AL_time']

        # # Reindex all files so indices follow properly
        # for key, val in self.variables.items():
        #     if key.startswith('test_') or key.startswith('val_') or (
        #             'rollout' in key):
        #         # Avoid saving all test and validation variables, and only
        #         # save rollout variables in rollout functions
        #         continue
        #     filename = str(key) + '.csv'
        #     file = pd.read_csv(os.path.join(self.results_folder, filename),
        #                        sep=',', header=None)
        #     file = file.drop(file.columns[0], axis=1)
        #     file.reset_index(drop=True)
        #     file.to_csv(os.path.join(self.results_folder, filename),
        #                 mode='w', header=False)

        if self.monitor_experiment:
            # Plot and save individual results of evaluation as csv and pdf
            if not self.ground_truth_approx:
                direct = True
            else:
                direct = False
            save_GP_data(self, direct=direct, verbose=self.verbose)
            plot_model_evaluation(
                self.grid_variables['Evaluation_grid'],
                self.grid_variables['Grid_controls'],
                self.grid_variables['Predicted_grid'],
                self.grid_variables['True_predicted_grid'],
                self.results_folder,
                ground_truth_approx=self.ground_truth_approx,
                l2_error_array=torch.mean(self.variables['l2_error_array'],
                                          dim=1), verbose=False)
            plot_GP(self, grid=torch.cat((
                self.grid_variables['Evaluation_grid'],
                self.grid_variables['Grid_controls']), dim=1),
                    verbose=self.verbose)

            # Plot computation time over time
            plt.close('all')
            name = 'Computation_time' + '.pdf'
            plt.plot(self.time[:, 0], self.time[:, 1], 'lime',
                     label='Time (s)')
            plt.title('Computation times over time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Computation time')
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            # Plot RMSE over time
            name = 'RMSE' + '.pdf'
            plt.plot(self.RMSE_time[:, 0], self.RMSE_time[:, 1], 'c',
                     label='RMSE')
            plt.title('RMSE between model and true dynamics over time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('RMSE')
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'SRMSE' + '.pdf'
            plt.plot(self.SRMSE_time[:, 0], self.SRMSE_time[:, 1], 'c',
                     label='SRMSE')
            plt.title('Standardized RMSE between model and true dynamics over '
                      'time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('SRMSE')
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'Average_log_likelihood' + '.pdf'
            plt.plot(self.log_likelihood_time[:, 0],
                     self.log_likelihood_time[:, 1], 'c', label='log_AL')
            plt.title(
                'Average log likelihood between model and true dynamics over '
                'time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Average log likelihood')
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'Standardized_average_log_likelihood' + '.pdf'
            plt.plot(self.stand_log_likelihood_time[:, 0],
                     self.stand_log_likelihood_time[:, 1], 'c', label='log_AL')
            plt.title(
                'Standardized average log likelihood between model and true '
                'dynamics over  time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Standardized average log likelihood')
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            # Plot heatmap of L2 error over grid on phase portraits
            # l2_error_array = torch.mean(self.variables['l2_error_array'], dim=1)
            # for i in range(self.grid.shape[1] - 1):
            #     name = 'Heatmap_L2_error' + str(i) + '.pdf'
            #     plt.scatter(self.grid.numpy()[:, i],
            #                 self.grid.numpy()[:, i + 1],
            #                 cmap='jet', c=l2_error_array)
            #     cbar = plt.colorbar()
            #     cbar.set_label('Squared L2 prediction error')
            #     plt.title('Prediction error over a grid')
            #     plt.xlabel('x_' + str(i))
            #     plt.ylabel('x_' + str(i + 1))
            #     plt.legend()
            #     plt.savefig(os.path.join(self.results_folder, name),
            #                 bbox_inches='tight')
            #     plt.close('all')
            # for i in range(self.grid_controls.shape[1] - 1):
            #     name = 'Heatmap_L2_error' + str(i + self.grid.shape[1] - 1) + \
            #            '.pdf'
            #     plt.scatter(self.grid_controls.numpy()[:, i],
            #                 self.grid_controls.numpy()[:, i + 1], cmap='jet',
            #                 c=l2_error_array)
            #     cbar = plt.colorbar()
            #     cbar.set_label('Squared L2 prediction error')
            #     plt.title('Prediction error over a grid')
            #     plt.xlabel('u_' + str(i))
            #     plt.ylabel('u_' + str(i + 1))
            #     plt.legend()
            #     plt.savefig(os.path.join(self.results_folder, name),
            #                 bbox_inches='tight')
            #     plt.close('all')
            # filename = 'l2_error_array.csv'
            # file = pd.DataFrame(self.variables['l2_error_array'].cpu().numpy())
            # file.to_csv(os.path.join(self.results_folder, filename),
            #             header=False)

            # Plot evolution of hyperparameters over time
            for i in range(1, self.hyperparams.shape[1]):
                name = 'Hyperparameter' + str(i) + '.pdf'
                plt.plot(self.hyperparams[:, 0], self.hyperparams[:, i],
                         c='darkblue', label='Hyperparameter')
                plt.title('Evolution of hyperparameters during exploration')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Hyperparameter value')
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.close('all')

            # Plot loss over iterations of hyperparameter optimization
            if 'loss' in self.variables.keys():
                name = 'Loss' + '.pdf'
                plt.plot(self.variables['loss'], '+-', label='loss')
                plt.title('Loss during hyperparameter optimization')
                plt.yscale('log')
                plt.legend()
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.close('all')

        if self.monitor_experiment and self.verbose:
            logging.info('L2 error per state')
            logging.info(self.specs['l2_error'])
            logging.info('Total RMSE (mean over grid)')
            logging.info(self.specs['RMSE'])
            logging.info('Total SRMSE (mean over grid)')
            logging.info(self.specs['SRMSE'])
            logging.info('Total log_AL (mean over grid)')
            logging.info(self.specs['log_AL'])
            logging.info('Total stand_log_AL (mean over grid)')
            logging.info(self.specs['stand_log_AL'])
            logging.info(
                'Rollout RMSE (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_RMSE'])
            logging.info(
                'Rollout SRMSE (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_SRMSE'])
            logging.info(
                'Rollout log_AL (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_stand_log_AL'])
            logging.info(
                'Rollout stand_log_AL (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_stand_log_AL'])

        logging.info('Final model:')
        if self.use_GPy:
            for i in range(len(self.model.kern.parameters)):
                logging.info(self.model.kern.parameters[i])
        else:
            logging.info(print_parameters(self.model))
        logging.info('Saved results in')
        logging.info(self.results_folder)
