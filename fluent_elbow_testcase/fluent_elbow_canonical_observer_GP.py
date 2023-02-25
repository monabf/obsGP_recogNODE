import logging
import os
import shutil
import sys

import dill as pkl
import gpytorch
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

sys.path.append('.')

from fluent_elbow_testcase.preprocess_data import find_nearest_point, \
    read_points, read_SVD_coefs, read_T_SVD_data, read_T_SVD_afterROM_data, \
    read_whole_field, compute_whole_field, form_Fluent_test_data, \
    approx_dynamics_from_data
from simulation.observers import dim1_observe_data, \
    MSM_justvelocity_observer_highgain_GP
from simulation.prior_means import read_discrete_GP_prior
from simulation.simulation_functions import form_GP_data, traj_from_data
from model_evaluation.plotting_functions import save_outside_data, plot_outside_data
from utils.utils import reshape_pt1, reshape_dim1, interpolate_func, reshape_dim1_tonormal
from utils.utils import start_log, stop_log
from utils.config import Config
from GP_models.GP_dyn_several_exp_datasets import GP_Dyn_Several_Exp_Datasets

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
# modes can be learned. Here the observer is a high-gain observer after
# transformation into the canonical coordinates.

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

    # General params
    config = Config(system='Continuous/Fluent_elbow/Discrete_model/'
                           'GP_justvelocity_highgain_observer_noisy_inputs',
                    data_folder='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/',
                    rom_folder='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/roms/new_good1_learn12_5modes_noSteadyInit/',
                    read_whole_field_data=False,
                    nb_SVD_modes=3,
                    limit_nb_SVD_modes=3,  # dx=3
                    train_scenarios=[2, 1],
                    test_scenarios=[1, 2, 3, 4],
                    nb_samples=10001,
                    t0_span=0,
                    tf_span=10000,
                    t0=0,
                    tf=10000,
                    hyperparam_optim='fixed_hyperparameters',
                    true_meas_noise_var=1e-10,
                    process_noise_var=0,
                    simu_solver='dopri5',
                    # solver_options={'rtol': 1e-8, 'atol': 1e-10},
                    nb_rollouts=0,  # no sense here since no true init_state
                    sliding_window_size=None,
                    sliding_window_size_each=None,  # avoid transient obs
                    verbose=False,
                    monitor_experiment=True,
                    multitask_GP=False,
                    sparse={'method': 'default', 'nb_inducing_inputs': 500},
                    # sparse=None,
                    memory_saving=False,
                    restart_on_loop=True,
                    GP_optim_method=torch.optim.Adam,  # SGD
                    GP_optim_training_iter=5,  # TODO
                    GP_optim_lr=1,
                    meas_noise_var=1e-3,
                    batch_adaptive_gain=None,
                    observer_prior_mean=None,
                    # prior_mean=MSM_continuous_to_discrete_justvelocity_prior_mean,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    prior_GP_model_file='../Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8/GP_model.pkl',
                    keep_prior_in_obs=False,
                    enrich_prior_GP=False,
                    derivative_function=None,
                    existing_results_folder=None,
                    fold_nb=0,
                    save_inside_fold=True,
                    full_rollouts=True,
                    plot_output=True)
    config.update(dict(nb_loops=len(config.train_scenarios), nb_subloops=1))
    whole_t_eval = np.linspace(config.t0, config.tf, config.nb_samples)

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
        df.to_pickle(config.data_folder + 'data.pkl')
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
        # # TODO numerical issues: normalize data?
        # for scenario in config.test_scenarios:
        #     traj = preprocessing.StandardScaler().fit_transform(data['out' + str(scenario)])
        #     data['out' + str(scenario)] = traj.copy()
        # Switch to pytorch
        for key, val in data.items():
            data[key] = torch.as_tensor(data[key])

    # System params
    if 'Continuous/Fluent_elbow' in config.system:
        config.update(dict(
            discrete=False,
            dt=config.dt,
            dt_before_subsampling=1,
            dynamics=None,
            controller=None,
            init_state=reshape_pt1(data['out2'][0, :]),
            init_state_estim=reshape_pt1(data['out2'][0, :]),
            init_control=reshape_pt1(data['exc2'][0, :])))
        if 'justvelocity_highgain_observer' in config.system:
            # High-gain observer in canonical coordinates (state = successive
            # time derivatives of output)
            config.update(dict(
                observer_function=MSM_justvelocity_observer_highgain_GP,
                discrete=False,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observation_matrix':  # SVD coefs of sensor
                                  torch.tensor([6.92140217e-03, -2.80060136e-03,
                                                -1.21308401e-02,
                                                -2.64675209e-05,
                                                5.56641843e-03]),
                              'test_observation_matrix':  # test SVD coefs
                                  torch.tensor([6.47743955e-03, -4.72619639e-03,
                                                -1.99883072e-03,
                                                -1.02860723e-03,
                                                -6.49726147e-05]),
                              # 'observer_gains': {'g': 2,
                              #                    'k1': 1.2,
                              #                    'k2': 0.669,
                              #                    'k3': 0.214,
                              #                    'k4': 0.0383,
                              #                    'k5': 0.0031}},
                              # 'observer_gains': {'g': 2,
                              #                    'k1': 0.544,
                              #                    'k2': 0.0987}},
                              'observer_gains': {'g': 2,
                                                 'k1': 0.764,
                                                 'k2': 0.243,
                                                 'k3': 0.031}},  # dx=3
                saturation=np.array([-1e2, 1e2]),
                observe_data=dim1_observe_data))
            config.update(dict(original_observe_data=lambda x: reshape_dim1(
                torch.matmul(reshape_pt1(x), config.prior_kwargs[
                    'observation_matrix'].t()))))
            # config.update(dict(init_state_estim=torch.cat((
            #         config.original_observe_data(config.init_state),
            #         torch.zeros((
            #             1, config.init_state.shape[1] - 1))), dim=1),
            #     test_observe_data=lambda x: reshape_dim1(
            #         torch.matmul(reshape_pt1(x), config.prior_kwargs[
            #                'test_observation_matrix'].t())))
            config.update(dict(init_state_estim=torch.cat((
                config.original_observe_data(config.init_state), torch.zeros((
                    1, 2))), dim=1),  # dx=3
                test_observe_data=lambda x: reshape_dim1(torch.matmul(
                    reshape_pt1(x), config.prior_kwargs[
                                        'test_observation_matrix'][:3].t()))))
        elif 'No_observer' in config.system:
            config.update(dict(prior_kwargs={'dt': config.dt,
                                             'dt_before_subsampling': 0.001},
                               observer_function=None,
                               observer_prior_mean=None,
                               observe_data=dim1_observe_data,
                               original_observe_data=dim1_observe_data))
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
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        likelihood.noise_covar.noise = torch.tensor([config.meas_noise_var])
        likelihood.noise = torch.tensor([1e-4])
        lengthscale_prior = gpytorch.priors.MultivariateNormalPrior(
            torch.tensor([150., 50., 5., 150., 150.]),  # dx=3
            torch.diag(torch.tensor([150., 50., 5., 150., 150.])))
        outputscale_prior = gpytorch.priors.NormalPrior(40., 40.)
        hypers = {'base_kernel.lengthscale': lengthscale_prior.mean,
                  'outputscale': outputscale_prior.mean}
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim,
                                       lengthscale_prior=lengthscale_prior),
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
                           observer=config.observer_function(config.cuda_device,
                                                             config)))
        if config.get('limit_nb_SVD_modes'):
            config.prior_kwargs[
                'limit_nb_SVD_modes'] = config.limit_nb_SVD_modes
        if config.prior_mean == read_discrete_GP_prior:
            hypers = {'base_kernel.lengthscale':
                          torch.tensor([150., 80., 90., 150., 150.]),
                      'outputscale': torch.tensor([10.])}  # dx=3
            prior_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=input_dim,
                                           lengthscale_prior=lengthscale_prior),
                outputscale_prior=outputscale_prior)
            prior_kernel.initialize(**hypers)
            config.update(dict(prior_kernel=prior_kernel))
            read_prior_GP = read_discrete_GP_prior(config.prior_GP_model_file,
                                                   config.prior_kernel)
            config.prior_kwargs['prior_GP_model'] = read_prior_GP.prior_GP_model
            config.update(dict(
                prior_mean=read_prior_GP.predict_onlymean,
                prior_mean_deriv=read_prior_GP.predict_deriv_onlymean,
                observer_prior_mean=read_prior_GP))
            print(config.prior_mean,
                  config.prior_kwargs['prior_GP_model'].kernel)
    else:
        raise Exception('Unknown system')

    # Loop to create data and learn GP over training scenarios
    for fold_nb in range(config.nb_loops):
        scenario = int(config.train_scenarios[fold_nb])

        # Form traj data for this fold
        xtraj = data['out' + str(scenario)][:10001, :]
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
            if 'justvelocity_highgain_observer' in config.system:
                # config.update(dict(init_state_estim=torch.cat((
                #     reshape_pt1(y_observed[0]), torch.zeros((
                #         1, config.init_state.shape[1] - 1))), dim=1)))
                config.update(dict(init_state_estim=torch.cat((
                    reshape_pt1(y_observed[0]), torch.zeros((
                        1, 2))), dim=1)))  # dx=3

            # Put all tensor data on cuda
            if torch.cuda.is_available():
                for key, val in config.items():
                    if torch.is_tensor(val):
                        val = val.cuda()
                xtraj, utraj, y_observed = xtraj.cuda(), utraj.cuda(), \
                                           y_observed.cuda()

            if fold_nb == 0:
                observer_prior_mean = config.observer_prior_mean
            elif (subloop == 0) and config.keep_prior_in_obs:
                # Always restart observer from prior since current GP has
                # never seen this training scenario before
                observer_prior_mean = config.observer_prior_mean
            else:
                observer_prior_mean = dyn_GP

            y_autom_diff = np.zeros((
                len(xtraj), config.init_state_estim.shape[1]))
            for i in range(config.init_state_estim.shape[1]):
                if i == 0:
                    y_autom_diff[:, i] = reshape_dim1_tonormal(
                        y_observed.numpy())  # No autom grad of signal in torch
                else:
                    y_autom_diff[:, i] = np.gradient(y_autom_diff[:, i - 1])
                    mean = np.mean(y_autom_diff[:, i])
                    std = np.std(y_autom_diff[:, i])
                    y_autom_diff[:, i] = np.clip(
                        y_autom_diff[:, i], a_min=mean - 3 * std,
                        a_max=mean + 3 * std)
            original_xtraj = xtraj.clone()
            xtraj = torch.as_tensor(y_autom_diff)

            xtraj_estim = traj_from_data(system=config.system,
                                         measurement=measurement,
                                         controller=controller,
                                         observer=config.observer, xtraj=xtraj,
                                         t_eval=t_eval, t0=config.t0,
                                         tf=config.tf,
                                         time=time,
                                         dt=config.dt_before_subsampling,
                                         meas_noise_var=config.true_meas_noise_var,
                                         init_control=config.init_control,
                                         init_state_estim=config.init_state_estim,
                                         method=config.simu_solver,
                                         dyn_config=config,
                                         GP=observer_prior_mean,
                                         discrete=config.discrete,
                                         verbose=config.verbose)
            # xtraj_estim = reshape_dim1(y_observed)  # dx=1
            xtraj_estim = xtraj_estim[:10001, :]
            # xtraj_estim = xtraj_estim + torch.normal(
            #     0, 1e-6, xtraj_estim.shape)  # TODO badly conditioned no noise?
            # for i in range(xtraj_estim.shape[1]):  # TODO
            #     plt.plot(xtraj[:, i])
            #     plt.plot(xtraj_estim[:, i])
            #     plt.show()

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
                config.update(dict(
                    init_state=config.init_state[:,
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
                new_results_folder = os.path.join(base_path,
                                                  'Loop_' + str(subloop))
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
                dyn_GP.evaluate_kalman_rollouts(
                    MSM_justvelocity_observer_highgain_GP, config.observe_data,
                    config.discrete)
                dyn_GP.evaluate_closedloop_rollouts(
                    MSM_justvelocity_observer_highgain_GP, config.observe_data)
            else:
                dyn_GP.evaluate_kalman_rollouts(config.observer,
                                                config.observe_data,
                                                config.discrete)
                dyn_GP.evaluate_closedloop_rollouts(config.observer,
                                                    config.observe_data)
            specs_file = os.path.join(
                dyn_GP.results_folder, 'Specifications.txt')
            with open(specs_file, 'a') as f:
                print('\n', file=f)
                print('This data was obtained by using an HGO directly on the '
                      'output for state estimation. This means the estimated '
                      'trajectory is in the canonical coordinates, and the '
                      '"true" trajectory is actually just a an automatic '
                      'differentiation of the true output and should not be '
                      'considered as the output!', file=f)

            # Also evaluate learned GP on fixed test (holdout) set
            X_test, U_test, Y_test, cut_idx_test = \
                form_Fluent_test_data(data, config.test_scenarios, config)
            original_Xtest = torch.as_tensor(X_test)
            original_Ytest = torch.as_tensor(Y_test)
            X_test = np.zeros((0, original_Xtest.shape[1]))
            Y_test = np.zeros((0, original_Ytest.shape[1]))
            for scenario in config.test_scenarios:
                x = data['out' + str(scenario)]
                test_observed = config.original_observe_data(x)
                test_autom_diff = np.zeros((len(x), x.shape[1]))
                # No autom grad of signal in torch, do it in numpy...
                for i in range(x.shape[1]):
                    if i == 0:
                        test_autom_diff[:, i] = reshape_dim1_tonormal(
                            test_observed)
                    else:
                        test_autom_diff[:, i] = np.gradient(
                            test_autom_diff[:, i - 1])
                        mean = np.mean(test_autom_diff[:, i])
                        std = np.std(test_autom_diff[:, i])
                        test_autom_diff[:, i] = np.clip(
                            test_autom_diff[:, i], a_min=mean - 3 * std,
                            a_max=mean + 3 * std)
                X_test = np.concatenate((X_test, reshape_dim1(
                    test_autom_diff[:-1, :])), axis=0)
                Y_test = np.concatenate((Y_test, reshape_dim1(
                    test_autom_diff[1:, :])), axis=0)
            X_test = torch.as_tensor(X_test)
            U_test = torch.as_tensor(U_test)
            Y_test = torch.as_tensor(Y_test)
            if config.get('limit_nb_SVD_modes'):
                # Test with only few SVD modes
                X_test = X_test[:, :config.limit_nb_SVD_modes]
                Y_test = Y_test[:, :config.limit_nb_SVD_modes]
            # X_test = reshape_dim1(config.original_observe_data(X_test.clone()))
            # X_test = X_test + torch.normal(
            #     # TODO too much for HGO?
            #     0, np.sqrt(config.true_meas_noise_var), X_test.shape)
            # Y_test = reshape_dim1(config.original_observe_data(Y_test.clone()))
            # Y_test = Y_test + torch.normal(
            #     # TODO too much for HGO?
            #     0, np.sqrt(config.true_meas_noise_var), Y_test.shape)  # dx=1
            print(X_test.shape, Y_test.shape, U_test.shape)
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
                    dyn_GP.evaluate_test_kalman_rollouts(
                        config.observer, config.observe_data, config.discrete,
                        no_GP_in_observer=True, only_prior=True)
                dyn_GP.step = old_step
                dyn_GP.sample_idx = old_sample_idx

            dyn_GP.evaluate_test_kalman_rollouts(config.observer,
                                                 config.observe_data,
                                                 config.discrete)
            dyn_GP.evaluate_test_closedloop_rollouts(config.observer,
                                                     config.observe_data)

            # # Predict temperature at another point in solid (not sensor)
            # for i in range(len(dyn_GP.test_rollout_list)):
            #     rollout_folder = os.path.join(
            #         dyn_GP.test_folder, 'Test_rollouts_' + str(dyn_GP.step - 1),
            #                             'Rollout_' + str(i))
            #     predicted_mean = pd.read_csv(os.path.join(
            #         rollout_folder, 'Predicted_mean_traj.csv'), sep=',',
            #         header=None)
            #     predicted_mean = predicted_mean.drop(
            #         predicted_mean.columns[0], axis=1).values
            #     true_mean = pd.read_csv(os.path.join(
            #         rollout_folder, 'True_traj.csv'), sep=',', header=None)
            #     true_mean = true_mean.drop(true_mean.columns[0], axis=1).values
            #     test_obs = config.test_observe_data(predicted_mean)
            #     true_test_obs = config.test_observe_data(true_mean)
            #     for k in range(y_observed.shape[1]):
            #         name = 'Rollout_test_output_predictions' + str(k) + '.pdf'
            #         plt.plot(time, true_test_obs[:, k], 'g',
            #                  label='Observed test output')
            #         plt.plot(time, test_obs[:, k],
            #                  label='Predicted test output', c='orange',
            #                  alpha=0.9)
            #         plt.title('Rollout of predicted and true test output over '
            #                   'time over testing data')
            #         plt.legend()
            #         plt.xlabel('Time steps')
            #         plt.ylabel('Output')
            #         plt.savefig(os.path.join(rollout_folder, name),
            #                     bbox_inches='tight')
            #         plt.close('all')

    # At end of training, save GP model with pickle
    with open(dyn_GP.results_folder + '/GP_submodel.pkl', 'wb') as f:
        pkl.dump(dyn_GP.model, f, protocol=4)
    with open(dyn_GP.results_folder + '/GP_model.pkl', 'wb') as f:
        pkl.dump(dyn_GP, f, protocol=4)

    stop_log()
