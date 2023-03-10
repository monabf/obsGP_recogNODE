import logging
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import os
import sys
import time as timep

import dill as pkl
import numpy as np
import pandas as pd
import pickle5 as pickle
import pytorch_lightning as pl
import seaborn as sb
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('.')

from simulation.controllers import Control_from_list
from simulation.observers import dim14_observe_data
from simulation.observer_functions import KKL, KKLu
from wandercraft_id_NN_models import MLPn, WDC_simple_recog4, \
    WDC_single_deformation_model, WDC_resmodel
from eval_WDC_data import plot_NODE_lin_rollouts
from NN_for_ODEs.learn_neural_ODE_several_exp_datasets import \
    Learn_NODE_difftraj_exp_datasets
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs, set_DF
from utils.utils import start_log, stop_log, reshape_pt1, reshape_dim1, \
    Interpolate_func
from utils.pytorch_utils import get_parameters
from utils.config import Config

sb.set_style("whitegrid")

# Script to learn NODEs from experimental Wandercraft data
# Learn residuals from prior linear model identified at Wandercraft

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

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TODO no GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'  # more explicit cuda errors

if __name__ == '__main__':
    start_log()
    start = timep.time()
    train = True  # TODO
    test = True
    colab = False
    dataset_on_GPU = False
    if torch.cuda.is_available():
        gpus = -1
        # strategy = pl.plugins.DDPPlugin(find_unused_parameters=False)
    else:
        gpus = 0
        # strategy = None
    print(gpus, torch.cuda.is_available())

    # Read data
    vars = dict.fromkeys(['u', 'theta', 'w'])
    dt = 1e-3  # TODO
    dt_before_subsampling = 1e-3
    data_folder = '../Data/Wandercraft_id/preprocessing/' \
                  '7-29-3difftraj_0.001subsampling_200samples_filter_gaussian50_butteru/'
    for key in vars.keys():
        data_file = data_folder + 'data_' + key + '.pkl'
        if colab:
            with open(data_file, "rb") as fh:
                data = pickle.load(fh)
                df = pd.DataFrame(data)
        else:
            df = pd.read_pickle(data_file).T
        vars[key] = torch.unsqueeze(torch.as_tensor(df.values), dim=-1)
    # nb_difftraj = df.shape[0]
    nb_samples = df.shape[1]
    t0 = 0.
    tf = t0 + (nb_samples - 1) * dt
    U_train = vars['u']
    X_train = torch.cat((vars['theta'], vars['w']), dim=-1)
    # Shorten nb of difftraj: get rid of some data to shorten computation
    nb_difftraj_max = int(np.min([600, len(X_train)]))  # TODO
    random_idx = torch.randperm(len(X_train))[:nb_difftraj_max]
    X_train = X_train[random_idx]
    U_train = U_train[random_idx]
    nb_difftraj = len(X_train)
    print(X_train.shape, U_train.shape, dt, tf)

    # General params
    config = Config(system='Continuous/Wandercraft_id/MLPn_noisy_inputs/'
                           'Residuals',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=nb_samples,
                    nb_difftraj=nb_difftraj,
                    t0_span=t0,
                    tf_span=tf,
                    t0=t0,
                    tf=tf,
                    dt=dt,
                    init_state_obs_method='KKL_u0T_back',
                    setD_method='butter_block_diag',
                    init_state_obs_T=100,  # TODO
                    # init_state_obs_Tu=1,
                    data_folder=data_folder,
                    NODE_file='../Figures/Continuous/Wandercraft_id'
                              '/MLPn_noisy_inputs/Residuals/7-29-3difftraj_0.001subsampling_200samples_filter_gaussian50/'
                              'y0T_u0T/Okish1_prior1e-6_decay1e-12_5x100_5x50_autograd_200samples_noise0.0_NODE_difftraj193_Adam0.005_y0T_u0T100/Learn_NODE.pkl',
                    true_meas_noise_var=0.,
                    process_noise_var=0.,
                    extra_noise=5e-5,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-8, 'atol': 1e-8},
                    scalerX_method='meanY_obs',
                    trainer_options={'max_epochs': 2000, 'gpus': gpus},
                    optim_method=torch.optim.Adam,
                    optim_lr=1e-2,
                    optim_minibatch_size=10,  # TODO
                    optim_shuffle=True,
                    optim_options={'weight_decay': 1e-12},
                    # l1_reg=1e-3,
                    prior_l2_reg=1e-6,
                    # KKL_l2_reg=1e-5,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.7, 'patience': 3,
                        'threshold': 0.01, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.001, patience=7,
                        verbose=False, mode='min'),
                    verbose=False,
                    monitor_experiment=False,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    continuous_model=False,
                    plot_output=True)

    # Train whole model
    if train:
        # Add noise scaled to var of signal to make more robust
        if config.extra_noise:
            # https://stackoverflow.com/questions/64074698/how-to-add-5-gaussian-noise-to-the-signal-data
            true_meas_noise_var_Xtrain = \
                config.extra_noise * torch.mean(
                    torch.var(X_train, dim=1), dim=0)
            true_meas_noise_var_Utrain = \
                config.extra_noise * torch.mean(
                    torch.var(U_train, dim=1), dim=0)
            config.update(dict(
                true_meas_noise_var_Xtrain=true_meas_noise_var_Xtrain,
                true_meas_noise_var_Utrain=true_meas_noise_var_Utrain))
            distrib_Xtrain = \
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(X_train.shape[-1]), torch.diag(
                        config.true_meas_noise_var_Xtrain))
            X_train += distrib_Xtrain.sample(X_train.shape[:-1])
            distrib_Utrain = \
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(U_train.shape[-1]), torch.diag(
                        config.true_meas_noise_var_Utrain))
            U_train += distrib_Utrain.sample(U_train.shape[:-1])
        init_state = torch.unsqueeze(torch.cat((
            reshape_dim1(X_train[:, 0, 0]),
            torch.zeros_like(reshape_dim1(X_train[:, 0, 0])),
            torch.zeros_like(reshape_dim1(X_train[:, 0, 1])),
            reshape_dim1(X_train[:, 0, 1])), dim=-1), dim=1)
        init_control = torch.unsqueeze(U_train[:, 0], dim=1)

        # System params
        config.update(dict(
            discrete=False,
            dt_before_subsampling=dt_before_subsampling,
            no_control=False,
            init_control=init_control,
            init_state=init_state,
            init_state_x=init_state.clone(),
            init_state_estim=init_state.clone(),
            n=init_state.shape[2],
            observe_data=dim14_observe_data,
            observe_data_x=dim14_observe_data,
            prior_kwargs={'dt': config.dt,
                          'dt_before_subsampling': config.dt}))
        if dataset_on_GPU:  # TODO
            X_train, U_train, init_state, init_control = \
                X_train.to(config.cuda_device), U_train.to(config.cuda_device),\
                init_state.to(config.cuda_device), \
                init_control.to(config.cuda_device)
        controller_list = []
        time = torch.arange(0., config.nb_samples * config.dt, config.dt,
                            device=U_train.device)
        for i in range(config.nb_difftraj):
            t_u = torch.cat((reshape_dim1(time), reshape_dim1(U_train[i])),
                            dim=1)
            control = Interpolate_func(
                x=t_u, t0=time[0], init_value=reshape_pt1(U_train[i, 0]))
            controller_list.append(control)
        controller = Control_from_list(controller_list, config.init_control)
        controller_args = [{} for i in range(len(controller_list))]
        config.update(dict(controller=controller,
                           controller_args=controller_args))

        # Create NN submodel of dynamics, then pack into Learn_NODE
        if config.no_control:
            nu_submodel = 0
            if config.init_state_obs_method == 'KKLu':
                logging.warning('KKLu without control: switching to '
                                'KKL_u0T to ignore the terms in u')
                config.init_state_obs_method = 'KKL_u0T'
        else:
            nu_submodel = config.init_control.shape[2]
        resmodel = MLPn(num_hl=5, n_in=config.n + nu_submodel,
                        n_hl=100, n_out=config.n, activation=nn.SiLU())  # TODO
        n_param, param = get_parameters(resmodel, verbose=True)
        submodel = WDC_resmodel(resmodel, config)
        config.update(dict(
            n_param=n_param, nu=config.init_control.shape[2],
            constrain_u=[-300, 300], constrain_x=[-0.5, 0.5],
            grid_inf=-0.5, grid_sup=0.5))

        # Recognition model to estimate x0 jointly with the dynamics
        # First define some configs for the inputs of the recognition model
        if 'KKL_u0T' in config.init_state_obs_method:
            dz = X_train.shape[2] * (config.n + 1)
            # dz += 90
            W0 = 2 * np.pi * 5
            D, F = set_DF(W0, dz, X_train.shape[-1], config.setD_method)
            z0 = torch.zeros(1, dz, device=X_train.device)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKL(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))
        elif 'KKLu' in config.init_state_obs_method:
            dw = 3  # to generate sinusoidal control with varying amplitude
            dz = (X_train.shape[2] + config.init_control.shape[1]) * (
                    config.n + dw + 1)
            # dz += 73
            W0 = 2 * np.pi * 5
            D, F = set_DF(W0, dz, X_train.shape[-1] +
                          config.init_control.shape[-1], config.setD_method)
            z0 = torch.zeros(1, dz, device=X_train.device)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'controller_args': config.controller_args,
                        'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKLu(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))

        # Define the inputs of the recognition model
        diff_init_state_obs = make_diff_init_state_obs(
            X_train, U_train, config.init_state_x, config.t_eval, config)

        # Define the actual recognition model (same for all init)
        if config.init_state_obs_method == 'fixed_recognition_model':
            init_state_model = WDC_simple_recog4(config.n, config.dt)
        else:
            init_state_model = MLPn(num_hl=5,
                                    n_in=torch.numel(diff_init_state_obs[0]),
                                    n_hl=50, n_out=config.n,
                                    activation=nn.SiLU())  # TODO
        _, X_train, U_train, config.t_eval = update_config_init_state_obs(
            diff_init_state_obs, init_state_model, X_train, X_train, U_train,
            config.t_eval, config)
        # Create Learn_NODE object
        NODE = Learn_NODE_difftraj_exp_datasets(
            X_train, U_train, submodel, config,
            sensitivity=config.sensitivity, dataset_on_GPU=dataset_on_GPU)
        # Train
        # To see logger in tensorboard, type in terminal then click on weblink:
        # tensorboard --logdir=../Figures/Continuous/Wandercraft_id/MLPn_noisy_inputs/ --port=8080
        logger = TensorBoardLogger(save_dir=NODE.results_folder + '/tb_logs')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              save_weights_only=True)
        if config.sensitivity == 'forward':
            traj_estim = NODE.train_forward_sensitivity()
        else:
            if config.optim_stopper:
                trainer = pl.Trainer(
                    callbacks=[config.optim_stopper, checkpoint_callback],
                    **config.trainer_options, logger=logger,
                    log_every_n_steps=1, check_val_every_n_epoch=2)
            else:
                trainer = pl.Trainer(
                    callbacks=[checkpoint_callback], **config.trainer_options,
                    logger=logger, log_every_n_steps=1,
                    check_val_every_n_epoch=2)
            trainer.fit(NODE)
        # Save and update estim x0
        NODE.save_model(checkpoint_path=checkpoint_callback.best_model_path)
    else:
        # # Recover from checkpoint: first need to recreate X_train, submodel...
        # checkpoint_path = '../Figures/Continuous/Wandercraft_id/MLPn_noisy_inputs/Residuals/' \
        #                   '29-30difftraj_0.005subsampling_40samples_filter_gaussian50/KKLu/0.48284310667867947_autograd_40samples_noise0.0_NODE_difftraj23_Adam0.001_KKLu20/tb_logs/default/version_0/checkpoints/epoch=549-step=549.ckpt'
        # checkpoint_model = torch.load(checkpoint_path)
        # print(checkpoint_model)
        # NODE = Learn_NODE_difftraj_exp_datasets(
        #     X_train, U_train, submodel, config,
        #     sensitivity=config.sensitivity, dataset_on_GPU=dataset_on_GPU)
        # NODE.load_state_dict(checkpoint_model['state_dict'])
        # NODE.results_folder = checkpoint_path.rsplit('/tb_logs', 1)[0]
        # config = NODE.config
        # NODE.save_model()

        # Or recover everything from pickle
        NODE = pkl.load(open(config.NODE_file, 'rb'))
        NODE.results_folder = config.NODE_file.rsplit('/', 1)[0]
        config = NODE.config

        # Compare with models identified at WDC
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True  # TODO
        vars = ['u_gaussian50', 'theta_gaussian50', 'w_gaussian50']
        dt = NODE.dt_before_subsampling
        n = len(NODE.rollout_list)
        # Save predictions over full data trajectories
        # plot_NODE_full_rollouts(NODE, dt, vars, verbose=verbose, save=save)
        # Compare rollouts: problem with initial condition, makes little sense!
        plot_NODE_lin_rollouts(n, NODE, NODE.rollout_list, lin_model=None,
                               verbose=verbose, save=save)
        # # Compare Bode diagrams: same procedure as for experimental data
        # # plot_NODE_lin_Bode(NODE, dt, lin_model, vars=vars, verbose=verbose,
        # #                    save=save, two_deformation=False)
        # Plot few training trajs
        os.makedirs(os.path.join(NODE.results_folder, 'Training_trajs'),
                    exist_ok=True)
        control_idx = NODE.train_val_idx[:n]
        xtraj_estim = NODE.NODE_model.forward_traj(
            NODE.init_state_estim[:n], NODE.controller[control_idx], NODE.t0,
            NODE.t_eval, NODE.init_control)
        y_observed = NODE.X_train[:n]
        y_pred = NODE.observe_data_x(xtraj_estim)
        print(xtraj_estim.shape, y_pred.shape, y_observed.shape,
              len(NODE.t_eval))
        import matplotlib.pyplot as plt

        for j in range(n):
            for i in range(y_pred.shape[2]):
                name = 'Training_trajs/y_pred' + str(j) + \
                       str(i) + '.pdf'
                plt.plot(y_observed[j, :, i], label='True')
                plt.plot(y_pred.detach()[j, :, i], label='Estimated')
                plt.title('Output')
                plt.legend()
                plt.savefig(os.path.join(NODE.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')

    if test:
        # Read test data
        vars = dict.fromkeys(['u', 'theta', 'w'])
        dt = 1e-3
        dt_before_subsampling = 1e-3
        data_folder = '../Data/Wandercraft_id/preprocessing/' \
                      '6-30-3+29difftraj_0.001subsampling_1000samples_filter_gaussian50_butteru/'
        for key in vars.keys():
            data_file = data_folder + 'data_' + key + '.pkl'
            if colab:
                with open(data_file, "rb") as fh:
                    data = pickle.load(fh)
                    df = pd.DataFrame(data)
            else:
                df = pd.read_pickle(data_file).T
            vars[key] = torch.unsqueeze(torch.as_tensor(df.values), dim=-1)
        # nb_difftraj = df.shape[0]
        nb_samples = df.shape[1]
        t0 = 0.
        tf = t0 + (nb_samples - 1) * dt
        U_test = vars['u']
        X_test = torch.cat((vars['theta'], vars['w']), dim=-1)
        # # Shorten nb of difftraj: get rid of some data to shorten computation
        # nb_difftraj_max = int(np.min([3, len(X_test)]))  # TODO
        # # random_idx = torch.randperm(len(X_test))[:nb_difftraj_max]
        # random_idx = torch.tensor([0, 10, 20])
        # X_test = X_test[random_idx, :500]
        # U_test = U_test[random_idx, :500]
        nb_rollouts = len(X_test)
        print(X_test.shape, U_test.shape, dt, tf)
        # Make init state obs for X_test, U_test
        init_state_x = torch.unsqueeze(torch.cat((
            reshape_dim1(X_test[:, 0, 0]),
            torch.zeros_like(reshape_dim1(X_test[:, 0, 0])),
            torch.zeros_like(reshape_dim1(X_test[:, 0, 1])),
            reshape_dim1(X_test[:, 0, 1])), dim=-1), dim=1)
        diff_init_state_obs = make_diff_init_state_obs(
            X_test, U_test, init_state_x, config.t_eval, config)
        _, X_test, U_test, config.t_eval = update_config_init_state_obs(
            diff_init_state_obs, NODE.init_state_model, X_test, X_test,
            U_test, config.t_eval, config)
        # Make test rollout data
        rollout_list = []
        i = 0
        while i < nb_rollouts:
            if NODE.init_state_model:
                # For ground_truth_approx, init_state in rollout_list
                # contains the inputs to the recognition model for Xtest
                # since it is known anyway, so that it can be used in NODE
                # rollouts directly
                init_state = reshape_pt1(diff_init_state_obs[i, 0])
            else:
                init_state = reshape_pt1(X_test[i, 0])
            control_traj = reshape_dim1(U_test[i])
            true_mean = reshape_dim1(X_test[i])
            rollout_list.append([init_state, control_traj, true_mean])
            i += 1
        NODE.step += 1
        NODE.evaluate_rollouts(rollout_list, plots=True)
        # Compare with WDC lin model
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        plot_NODE_lin_rollouts(nb_rollouts, NODE, rollout_list,
                               lin_model=lin_model,
                               verbose=verbose, save=save)
        # Also run these test rollouts with EKF
        C = torch.tensor([[1, 0, 0., 0],
                          [0, 0, 0, 1.]])
        meas = torch.tensor([1e-3, 1e1])
        process = torch.tensor([1e-3, 1e-2, 1e-1, 1e0])
        init = torch.tensor([1e-3, 1e-2, 1e-1, 1e0])
        NODE.config.update(
            {'prior_kwargs': {'n': NODE.n, 'observation_matrix': C,
                              'EKF_meas_covar': meas * torch.eye(NODE.p),
                              'EKF_process_covar': process * torch.eye(NODE.n),
                              'EKF_init_covar': init * torch.eye(NODE.n),
                              'EKF_added_meas_noise_var': 0.}})
        NODE.evaluate_EKF_rollouts(rollout_list, plots=True)
        # Compare with WDC lin model
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        vars = ['u_gaussian50', 'theta_gaussian50', 'w_gaussian50']
        dt = NODE.dt_before_subsampling
        # Compare EKF rollouts
        plot_NODE_lin_rollouts(nb_rollouts, NODE, rollout_list,
                               rollouts_title='EKF_rollouts',
                               lin_model=lin_model, type='EKF',
                               verbose=verbose, save=save)
        specs_file = os.path.join(NODE.results_folder, 'Specifications.txt')
        with open(specs_file, 'a') as f:
            print('\n', file=f)
            print('\n', file=f)
            print('\n', file=f)
            print(f'Test rollouts: {data_folder}', file=f)
            print(f'Number of test rollouts: {nb_rollouts}', file=f)
            print(f'Prior kwargs for rollouts (including EKF): '
                  f'{NODE.config.prior_kwargs}', file=f)
            for key, val in NODE.specs.items():
                if 'rollout' in key:
                    print(key, ': ', val, file=f)

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
