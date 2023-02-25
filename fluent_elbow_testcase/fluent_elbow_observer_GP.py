import logging
import os
import shutil
import sys
import time as timep

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TODO no GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'  # more explicit cuda errors

import dill as pkl
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import GPy
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

sys.path.append('.')

from GP_models.GP_dyn_several_exp_datasets import GP_Dyn_Several_Exp_Datasets
from fluent_elbow_testcase.preprocess_data import find_nearest_point, \
    read_points, read_SVD_coefs, read_T_SVD_data, read_T_SVD_afterROM_data, \
    read_whole_field, compute_whole_field, form_Fluent_test_data, \
    approx_dynamics_from_data
from simulation.observers import MSM_justvelocity_observer_highgain_GP, \
    EKF_GP, dim1_observe_data
from simulation.prior_means import read_discrete_GP_prior
from simulation.simulation_functions import form_GP_data, traj_from_data
from model_evaluation.plotting_functions import save_outside_data, \
    plot_outside_data
from utils.utils import reshape_pt1, reshape_dim1, start_log, \
    stop_log, interpolate_func
from utils.config import Config

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

# Script to learn dynamics of SVD modes of the temperature evolution in the
# elbow pipe (as simulated by Fluent) with a GP. Measurement of temperature
# only at one sensor, so that the SVD modes need to be estimated from the
# temperature at one point with an observer before the dynamics of these
# modes can be learned. Here the observer is an Extended Kalman Filter.

# Logging
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(
            '../Figures/Logs', 'log' + str(sys.argv[1]))),
        logging.StreamHandler(sys.stdout)
    ])


def true_dynamics(x, control):
    # Needs to be redefined outside of main for unpickling GPy model, so only
    # with pickled GP prior!!!
    # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    # https://www.stefaanlippens.net/python-pickling-and-dealing-with-attributeerror-module-object-has-no-attribute-thing.html
    return approx_dynamics_from_data(x, control, X, U, Y)


if __name__ == '__main__':
    start_log()
    start = timep.time()

    # General params
    config = Config(system='Continuous/Fluent_elbow/Discrete_model/'
                           'GP_EKF_observer_noisy_inputs',
                    data_folder='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/',
                    rom_folder='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/roms/new_good1_learn12_5modes_noSteadyInit/',
                    use_GPy=True,
                    read_whole_field_data=False,
                    nb_SVD_modes=5,
                    limit_nb_SVD_modes=5,  # dx=3
                    train_scenarios=[2, 1],
                    test_scenarios=[1, 2, 3, 4],
                    nb_samples=10001,
                    t0_span=0.,
                    tf_span=10000.,
                    t0=0.,
                    tf=10000.,
                    hyperparam_optim='fixed_hyperparameters',
                    true_meas_noise_var=0.00255366,
                    process_noise_var=0,
                    meas_noise_var=1e-6,
                    simu_solver='dopri5',
                    nb_rollouts=1,
                    rollout_length=10,
                    sliding_window_size=None,  # TODO
                    sliding_window_size_each=None,  # avoid transient EKF
                    backwards_after_estim=False,  # solve xhat backwards
                    verbose=False,
                    monitor_experiment=True,
                    memory_saving=False,
                    restart_on_loop=True,
                    sparse={'method': 'VarDTC', 'nb_inducing_inputs': 500},
                    GP_optim_method=torch.optim.SGD,
                    GP_optim_training_iter=0,
                    GP_optim_lr=1e-2,
                    batch_adaptive_gain=None,
                    observer_prior_mean=read_discrete_GP_prior,
                    prior_mean=read_discrete_GP_prior,
                    prior_mean_deriv=None,
                    prior_GP_model_file='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/GP_model_manuscript2.pkl',
                    keep_prior_in_obs=False,  # TODO
                    enrich_prior_GP=False,  # TODO
                    derivative_function=None,
                    existing_results_folder=None,
                    fold_nb=0,
                    save_inside_fold=True,
                    full_rollouts=True,
                    plot_output=True)
    config.update(dict(nb_loops=len(config.train_scenarios), nb_subloops=1))
    whole_t_eval = torch.linspace(config.t0, config.tf, config.nb_samples)
    if config.enrich_prior_GP and config.get('limit_nb_SVD_modes'):
        logging.warning('You are trying to enrich a previous GP model while '
                        'limiting the dimension of the considered data. Make '
                        'sure the previous and new models are compatible.')

    if config.read_whole_field_data:
        # Read whole data from Fluent simulations + DynaROM SVD
        points = read_points(config.data_folder)
        whole_field_original = read_whole_field(config.data_folder, len(points),
                                                config.nb_loops)
        T_SVD = read_T_SVD_data(config.rom_folder, config.nb_loops)

        # Pickle the data for further use
        f = open(config.data_folder + 'data.pkl', 'wb')
        data = T_SVD.copy()
        data.update({'time': reshape_dim1(np.arange(len(T_SVD['out1'])))})
        for i in range(1, config.nb_loops + 1):
            data.update({'exc' + str(i): whole_field_original['exc' + str(i)]})
        df = pd.DataFrame().append(data, ignore_index=True)
        df.to_pickle(config.data_folder + 'data.pkl', protocol=4)
        if config.verbose:
            # Make more plots of original data, takes extra time
            sensor = np.array([0.9679914, -0.1128347, -0.2924094])
            sensor, sensor_idx = find_nearest_point(sensor, points)
            T_SVD_afterROM = read_T_SVD_afterROM_data(config.rom_folder,
                                                      config.nb_loops)
            SVD_coefs = read_SVD_coefs(config.rom_folder, config.nb_SVD_modes)
            whole_field_afterSVD = compute_whole_field(T_SVD, SVD_coefs,
                                                       config.nb_loops)
            sensor_SVD_coefs = SVD_coefs[sensor_idx]
            # Plot data (before and after SVD) at one point in field
            plt.plot(whole_field_original['out1'][:, -10000])
            plt.plot(whole_field_afterSVD['out1'][:, -10000])
            plt.plot(whole_field_original['out2'][:, -10000])
            plt.plot(whole_field_afterSVD['out2'][:, -10000])
            plt.plot(whole_field_original['out3'][:, -10000])
            plt.plot(whole_field_afterSVD['out3'][:, -10000])
            plt.plot(whole_field_original['out4'][:, -10000])
            plt.plot(whole_field_afterSVD['out4'][:, -10000])
            plt.show()
            plt.close('all')
        # Switch to pytorch
        for key, val in data.items():
            data[key] = torch.as_tensor(data[key])
    else:
        # Read data directly from pickle
        df = pd.read_pickle(config.data_folder + 'data.pkl')
        df = df.iloc[0]
        data = df.to_dict()
        # Switch to pytorch
        for key, val in data.items():
            data[key] = torch.as_tensor(data[key])

    # System params
    if 'Continuous/Fluent_elbow' in config.system:
        config.update(dict(
            discrete=False,
            dt=config.dt,
            dt_before_subsampling=config.dt,
            dynamics=None,
            controller=None,
            init_state=reshape_pt1(data['out2'][0, :]),  # dx=3
            init_state_estim=reshape_pt1(data['out2'][0, :]),
            init_control=reshape_pt1(data['exc2'][0, :])))
        if 'GP_EKF' in config.system:
            # EKF knowing prior dynamics model
            EKF_process_covar = torch.tensor(
                [[1e3, 1e-3, 1e-3, 1e-3, 1e-3]]).T * \
                                torch.eye(config.init_state.shape[1])
            # EKF_process_covar = torch.tensor(  # TODO
            #     [[1e3, 1e-3, 1e-3, 1e-3, 1e-2]]).T * \
            #                     torch.eye(config.init_state.shape[1])
            # EKF_process_covar = torch.tensor([[1e3, 1e-3, 1e-3]]).T * \
            #                     torch.eye(config.init_state.shape[1])  # dx=3
            EKF_meas_covar = torch.eye(1) * 1e-6
            config.update(dict(
                observer_function=EKF_GP,
                discrete=True,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 1,
                              'n': config.init_state.shape[1],
                              'original_observation_matrix':
                                  torch.tensor([6.92140217e-03, -2.80060136e-03,
                                                -1.21308401e-02,
                                                -2.64675209e-05,
                                                5.56641843e-03][:]),  # dx=3
                              'observation_matrix':  # SVD coefs of sensor
                                  torch.tensor([6.92140217e-03, -2.80060136e-03,
                                                -1.21308401e-02,
                                                -2.64675209e-05,
                                                5.56641843e-03],
                                               device=config['cuda_device'][:]),
                              'test_observation_matrix':  # test SVD coefs
                                  torch.tensor([6.47743955e-03, -4.72619639e-03,
                                                -1.99883072e-03,
                                                -1.02860723e-03,
                                                -6.49726147e-05],
                                               device=config['cuda_device'][:]),
                              'EKF_process_covar': EKF_process_covar,
                              'EKF_meas_covar': EKF_meas_covar}))
            # saturation=np.array([-1e8, 1e8])))
            config.update(dict(init_state_estim=torch.cat((
                config.init_state, reshape_pt1(torch.flatten(
                    1e-3 * torch.eye(config.init_state.shape[1])))), dim=1),
                original_observe_data=lambda x: reshape_dim1(torch.matmul(
                    reshape_pt1(x), config.prior_kwargs[
                        'original_observation_matrix'].t())),
                observe_data=lambda x: reshape_dim1(torch.matmul(
                    reshape_pt1(x), config.prior_kwargs[
                        'observation_matrix'].t())),
                test_observe_data=lambda x: reshape_dim1(torch.matmul(
                    reshape_pt1(x), config.prior_kwargs[
                        'test_observation_matrix'].t()))))
        elif 'No_observer' in config.system:
            config.update(dict(prior_kwargs={'dt': config.dt,
                                             'dt_before_subsampling': 1},
                               observer_function=None,
                               observer_prior_mean=None,
                               observe_data=dim1_observe_data,
                               original_observe_data=dim1_observe_data,
                               test_observe_data=dim1_observe_data))
        # Create kernel
        if config.limit_nb_SVD_modes:
            nb_SVD = config.limit_nb_SVD_modes  # dx=3
        else:
            nb_SVD = config.nb_SVD_modes
        if config.get('gamma') == 0:
            input_dim = nb_SVD
            no_control = True
        else:
            input_dim = nb_SVD + config.init_control.shape[1]
            no_control = False
        if config.get('limit_nb_SVD_modes'):
            logging.info('Adapt kernel input dimension and lengthscales to '
                         'number of SVD modes used')
        # For each part of the likelihood (Gaussian noise, scale kernel,
        # RBF kernel), define a prior distribution which will influence the
        # hyperparameter optimization, and an initial value
        if config.sparse and (config.sparse.get('method') == 'SparseGP'):
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([nb_SVD]),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10))
            likelihood.noise_covar.noise = torch.tensor([config.meas_noise_var])
            likelihood.noise = torch.tensor([config.meas_noise_var])
        else:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=nb_SVD,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10))
            likelihood.noise = 1e-10
            likelihood.task_noises = torch.tensor(
                [config.meas_noise_var] * nb_SVD)
            likelihood.register_prior(
                'noise_prior',
                gpytorch.priors.MultivariateNormalPrior(torch.tensor(
                    [config.meas_noise_var] * nb_SVD),
                    torch.diag(torch.tensor([100.] * nb_SVD))),
                'noise')
        if config.num_latents:
            logging.info('MultitaskGP: kernel batch shape is num_latents')
            nb_SVD = config.num_latents
        lengthscale_prior = gpytorch.priors.MultivariateNormalPrior(
            # torch.tensor([150., 80., 90., 90., 90., 150., 150.]),  # dx=3
            torch.tensor([10., 150., 150., 150., 150., 100., 150.]),  # dx=3
            torch.diag(torch.tensor([150., 80., 90., 90., 90., 150., 150.])))
        outputscale_prior = gpytorch.priors.NormalPrior(1., 1.)  # TODO
        hypers = {'base_kernel.lengthscale': lengthscale_prior.mean,
                  'outputscale': outputscale_prior.mean}
        if config.use_GPy:
            kernel = GPy.kern.RBF(
                input_dim=input_dim,
                variance=outputscale_prior.mean.cpu().numpy(),
                lengthscale=lengthscale_prior.mean.cpu().numpy(), ARD=True)
            kernel.unconstrain()
            kernel.variance.set_prior(GPy.priors.Gaussian(1., 1.))
            kernel.lengthscale.set_prior(
                GPy.priors.MultivariateGaussian(
                    np.array([10., 150., 150., 150., 150., 100., 150.]),
                    np.diag([10., 150., 150., 150., 150., 100., 150.])))
        else:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([nb_SVD]),
                    ard_num_dims=input_dim,
                    lengthscale_prior=lengthscale_prior),
                batch_shape=torch.Size([nb_SVD]),
                outputscale_prior=outputscale_prior)
            kernel.initialize(**hypers)

        # Update config
        config.update(dict(constrain_u=[],
                           constrain_x=[],
                           grid_inf=[],
                           grid_sup=[],
                           input_dim=input_dim,
                           no_control=no_control,
                           kernel=kernel,
                           likelihood=likelihood,
                           observer=config.observer_function(
                               config.cuda_device, config)))

        if config.get('limit_nb_SVD_modes'):
            config.prior_kwargs[
                'limit_nb_SVD_modes'] = config.limit_nb_SVD_modes

        if (config.observer_prior_mean == read_discrete_GP_prior) or (
                config.prior_mean == read_discrete_GP_prior):
            lengthscale_prior = gpytorch.priors.MultivariateNormalPrior(
                torch.tensor([150., 80., 90., 90., 90., 150., 150.]),  # dx=3
                torch.diag(torch.tensor([10.] * input_dim)))
            outputscale_prior = gpytorch.priors.NormalPrior(10., 10.)
            hypers = {'base_kernel.lengthscale': lengthscale_prior.mean,
                      'outputscale': outputscale_prior.mean}
            if config.use_GPy:
                prior_kernel = GPy.kern.RBF(
                    input_dim=input_dim,
                    variance=outputscale_prior.mean.cpu().numpy(),
                    lengthscale=lengthscale_prior.mean.cpu().numpy(), ARD=True)
            else:
                prior_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        batch_shape=torch.Size([nb_SVD]),
                        ard_num_dims=input_dim,
                    ),  # lengthscale_prior=lengthscale_prior),
                    batch_shape=torch.Size([nb_SVD]),
                )  # outputscale_prior=outputscale_prior)
                prior_kernel.initialize(**hypers)
            read_prior_GP = read_discrete_GP_prior(
                config.prior_GP_model_file, prior_kernel)
            config.prior_kwargs['prior_GP_model'] = read_prior_GP.prior_GP_model
            if config.observer_prior_mean == read_discrete_GP_prior:
                config.update(dict(observer_prior_mean=read_prior_GP))
            if config.prior_mean == read_discrete_GP_prior:
                config.update(dict(
                    prior_mean=read_prior_GP.predict_onlymean,
                    prior_mean_deriv=read_prior_GP.predict_deriv_onlymean))
            print(config.prior_mean,
                  config.prior_kwargs['prior_GP_model'].kernel)
    else:
        raise Exception('Unknown system')

    # Loop to create data and learn GP over training scenarios
    for fold_nb in range(config.nb_loops):
        scenario = int(config.train_scenarios[fold_nb])

        # Form traj data for this fold
        xtraj = data['out' + str(scenario)][:10001, :]  # dx=3
        utraj = data['exc' + str(scenario)][:10001, :]
        y_observed = config.original_observe_data(xtraj)
        y_observed = y_observed + torch.normal(0, np.sqrt(
            config.true_meas_noise_var), size=y_observed.shape)
        time = reshape_dim1(data['time'])[:10001, :]
        t_x = torch.cat((time, xtraj), dim=1)
        t_u = torch.cat((time, utraj), dim=1)
        t_y = torch.cat((time, y_observed), dim=1)
        # Put all tensor data on cuda
        if torch.cuda.is_available():
            t_x, t_u, t_y = t_x.cuda(), t_u.cuda(), t_y.cuda()

        # Returning u(t) and y(t) interpolated from data
        controller = interpolate_func(x=t_u, t0=config.t0,
                                      init_value=config.init_control)
        measurement = interpolate_func(x=t_y, t0=config.t0,
                                       init_value=y_observed[0])

        if not controller or (len(torch.nonzero(utraj, as_tuple=False)) == 0):
            config.update(dict(no_control=True))
        else:
            config.update(dict(no_control=False))
        # Prepare to cut data in subloops
        whole_xtraj = xtraj.clone()
        whole_utraj = utraj.clone()
        whole_y_observed = y_observed.clone()
        whole_time = time.clone()
        N = int(np.ceil(len(whole_xtraj) / config.nb_subloops))

        # Make loop inside each training scenario
        for subloop in range(config.nb_subloops):
            # Form estimated data with observer from measurement
            xtraj = reshape_dim1(whole_xtraj[subloop * N:(subloop + 1) * N])
            utraj = reshape_dim1(whole_utraj[subloop * N:(subloop + 1) * N])
            y_observed = reshape_dim1(
                whole_y_observed[subloop * N:(subloop + 1) * N])
            time = reshape_dim1(whole_time[subloop * N:(subloop + 1) * N])
            t_eval = whole_t_eval[subloop * N:(subloop + 1) * N]
            config.update(dict(fold_nb=fold_nb,
                               init_state=reshape_pt1(xtraj[0, :]),
                               init_control=reshape_pt1(utraj[0, :])))
            if 'GP_EKF' in config.system:
                n = config.init_state.shape[1]
                init_state_estim = torch.cat((
                    config.init_state,  # TODO
                    # torch.zeros(1, n),
                    # config.init_state + 500. * torch.ones_like(
                    #     config.init_state),
                    # reshape_pt1(torch.mean(whole_xtraj, dim=0)),
                    reshape_pt1(torch.flatten(
                        config.prior_kwargs['EKF_process_covar']))), dim=1)
                config.update(dict(init_state_estim=init_state_estim))

            if fold_nb == 0:
                observer_prior_mean = config.observer_prior_mean
            elif (subloop == 0) and config.keep_prior_in_obs:
                # Restart observer from prior since current GP has
                # never seen this training scenario before
                observer_prior_mean = config.observer_prior_mean
            else:
                observer_prior_mean = dyn_GP
            xtraj_estim = traj_from_data(system=config.system,
                                         measurement=measurement,
                                         controller=controller,
                                         observer=config.observer, xtraj=xtraj,
                                         t_eval=t_eval, t0=config.t0,
                                         tf=config.tf, time=time,
                                         dt=config.dt_before_subsampling,
                                         meas_noise_var=config.true_meas_noise_var,
                                         init_control=config.init_control,
                                         init_state_estim=config.init_state_estim,
                                         method=config.simu_solver,
                                         dyn_config=config,
                                         GP=observer_prior_mean,
                                         discrete=config.discrete,
                                         verbose=config.verbose)

            # Create data for GP, noiseless or noisy X, noiseless U, noisy Y
            X, U, Y = form_GP_data(system=config.system, xtraj=xtraj,
                                   xtraj_estim=xtraj_estim, utraj=utraj,
                                   meas_noise_var=config.true_meas_noise_var,
                                   y_observed=y_observed,
                                   derivative_function=config.derivative_function,
                                   model=observer_prior_mean)
            if config.get('limit_nb_SVD_modes'):
                # Test with only few SVD modes
                X = X[:, :config.limit_nb_SVD_modes]
                Y = Y[:, :config.limit_nb_SVD_modes]
                config.update(dict(init_state=config.init_state[:,
                                              :config.limit_nb_SVD_modes]))

            # Define true dynamics, controller, measurement from data
            if 'Fluent_elbow' in config.system:
                def true_dynamics(x, control):
                    return approx_dynamics_from_data(x, control, X, U, Y)
            else:
                raise Exception('No ground truth has been defined.')
            config.update(dict(true_dynamics=true_dynamics))

            # Initialize or re-initialize GP
            if (fold_nb == 0) and (subloop == 0):
                if config.enrich_prior_GP:
                    old_X, old_U, old_Y = \
                        config.prior_kwargs['prior_GP_model'].X, \
                        config.prior_kwargs['prior_GP_model'].U, \
                        config.prior_kwargs['prior_GP_model'].Y
                    dyn_GP = GP_Dyn_Several_Exp_Datasets(old_X, old_U, old_Y,
                                                         config)
                    dyn_GP.model = config.prior_kwargs['prior_GP_model'].model
                    config.existing_results_folder = os.path.normpath(
                        dyn_GP.results_folder).split('Crossval_Fold')[0]
                else:
                    dyn_GP = GP_Dyn_Several_Exp_Datasets(X, U, Y, config)
                    config.existing_results_folder = os.path.normpath(
                        dyn_GP.results_folder).split('Crossval_Fold')[0]
            elif subloop == 0:
                (base_path, previous_loop) = os.path.split(
                    dyn_GP.results_folder)
                (base_path, previous_fold) = os.path.split(base_path)
                new_results_folder = os.path.join(
                    base_path, 'Crossval_Fold_' + str(fold_nb),
                               'Loop_' + str(subloop))
                os.makedirs(new_results_folder, exist_ok=False)
                dyn_GP.set_results_folder(new_results_folder)
                dyn_GP.set_config(config)
            else:
                (base_path, previous_loop) = os.path.split(
                    dyn_GP.results_folder)
                new_results_folder = os.path.join(
                    base_path, 'Loop_' + str(subloop))
                os.makedirs(new_results_folder, exist_ok=False)
                dyn_GP.set_results_folder(new_results_folder)
                dyn_GP.set_config(config)

            # Set up outside data to save
            data_to_save = {'xtraj': xtraj, 'xtraj_estim': xtraj_estim,
                            'y_obs': y_observed}
            if config.batch_adaptive_gain:
                gain_time = torch.tensor(
                    config.prior_kwargs['observer_gains']['g'])
                data_to_save.update({'gain_time': gain_time})
            elif 'adaptive' in config.system:
                output_error = reshape_dim1(
                    torch.square(xtraj[:, 0] - xtraj_estim[:, 0]))
                if subloop == 0:
                    gain_time = reshape_dim1(xtraj_estim[:, -1])
                else:
                    gain_time = torch.cat((
                        gain_time, reshape_dim1(xtraj_estim[:, -1])))
                data_to_save.update(
                    {'gain_time': gain_time, 'output_error': output_error})

            # Train GP with estimated data, then save and evaluate results
            if config.monitor_experiment:
                save_outside_data(dyn_GP, data_to_save)
                plot_outside_data(dyn_GP, data_to_save)
            if (fold_nb == 0) and (subloop == 0):
                if config.enrich_prior_GP:
                    dyn_GP.learn(new_X=X, new_Y=Y, new_U=U)
                else:
                    dyn_GP.learn()
            else:
                dyn_GP.learn(new_X=X, new_Y=Y, new_U=U)

            dyn_GP.save()
            if 'justvelocity_adaptive' in config.system:
                # Do not adapt observer gains for closed-loop rollouts
                dyn_GP.evaluate_closedloop_rollouts(
                    MSM_justvelocity_observer_highgain_GP, config.observe_data)
            else:
                dyn_GP.evaluate_closedloop_rollouts(config.observer,
                                                    config.observe_data)

            # Also evaluate learned GP on fixed test (holdout) set
            X_test, U_test, Y_test, cut_idx_test = \
                form_Fluent_test_data(data, config.test_scenarios, config)
            X_test = torch.as_tensor(X_test)
            U_test = torch.as_tensor(U_test)
            Y_test = torch.as_tensor(Y_test)
            if config.get('limit_nb_SVD_modes'):
                # Test with only few SVD modes
                X_test = X_test[:, :config.limit_nb_SVD_modes]
                Y_test = Y_test[:, :config.limit_nb_SVD_modes]
            dyn_GP.test(X_test, U_test, Y_test, cut_idx_test)

            if (fold_nb == 0) and (subloop == 0):
                # Run rollouts using only priors, before learning (step=-1)
                rollouts_folder = os.path.join(dyn_GP.test_folder,
                                               'Test_rollouts_0')
                new_rollouts_folder = os.path.join(dyn_GP.test_folder,
                                                   'Test_rollouts_-1')
                shutil.copytree(rollouts_folder, new_rollouts_folder,
                                ignore=shutil.ignore_patterns(
                                    '*.pdf', 'Predicted*'))
                old_step, dyn_GP.step = dyn_GP.step, 0
                old_sample_idx, dyn_GP.sample_idx = dyn_GP.sample_idx, 0
                dyn_GP.evaluate_test_closedloop_rollouts(
                    config.observer, config.observe_data,
                    no_GP_in_observer=True)
                if config.prior_mean:
                    # Also run open-loop rollouts with prior before learning
                    dyn_GP.evaluate_test_rollouts(only_prior=True)
                    # dyn_GP.evaluate_test_kalman_rollouts(
                    #     config.observer, config.observe_data, config.discrete,
                    #     no_GP_in_observer=True, only_prior=True)  # TODO
                dyn_GP.step = old_step
                dyn_GP.sample_idx = old_sample_idx

            # dyn_GP.evaluate_test_kalman_rollouts(config.observer,
            #                                      config.observe_data,
            #                                      config.discrete)
            dyn_GP.evaluate_test_closedloop_rollouts(config.observer,
                                                     config.observe_data)

            # Predict temperature at another point in solid (not sensor)
            for i in range(len(dyn_GP.test_rollout_list)):
                rollout_folder = os.path.join(
                    dyn_GP.test_folder, 'Test_rollouts_' + str(dyn_GP.step - 1),
                                        'Rollout_' + str(i))
                predicted_mean = pd.read_csv(os.path.join(
                    rollout_folder, 'Predicted_mean_traj.csv'), sep=',',
                    header=None)
                predicted_mean = torch.as_tensor(predicted_mean.drop(
                    predicted_mean.columns[0], axis=1).values)
                true_mean = pd.read_csv(os.path.join(
                    rollout_folder, 'True_traj.csv'), sep=',', header=None)
                true_mean = torch.as_tensor(true_mean.drop(
                    true_mean.columns[0], axis=1).values)
                test_obs = config.test_observe_data(predicted_mean)
                true_test_obs = config.test_observe_data(true_mean)
                for k in range(y_observed.shape[1]):
                    name = 'Rollout_test_output_predictions' + str(k) + '.pdf'
                    plt.plot(true_test_obs[:, k], 'g',
                             label='Observed test output')
                    plt.plot(test_obs[:, k], label='Predicted test output',
                             c='orange', alpha=0.9)
                    plt.title('Rollout of predicted and true test output over '
                              'time over testing data')
                    plt.legend()
                    plt.xlabel('Time steps')
                    plt.ylabel('Output')
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                    plt.close('all')

        # At end of training, save GP model with pickle
        with open(dyn_GP.results_folder + '/GP_submodel.pkl', 'wb') as f:
            pkl.dump(dyn_GP.model, f, protocol=4)
        with open(dyn_GP.results_folder + '/GP_model.pkl', 'wb') as f:
            pkl.dump(dyn_GP, f, protocol=4)

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
