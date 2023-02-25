import logging
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import os
import sys
import time as timep

import dill as pkl
import numpy as np
import pytorch_lightning as pl
import scipy
import seaborn as sb
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('.')

from simulation.controllers import sin_controller_1D
from simulation.dynamics_functions import Reverse_Duffing
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data
from simulation.observer_functions import KKL, KKLu, \
    MSM_justvelocity_observer_highgain_GP
from revDuff_NN_models import MLP2
from NN_for_ODEs.learn_neural_ODE import Learn_NODE
from NN_for_ODEs.NODE_utils import make_init_state_obs, \
    update_config_init_state_obs, set_DF
from utils.utils import start_log, stop_log
from utils.pytorch_utils import get_parameters
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

# Script to learn physical dynamics (backPhi) from canonical observations
# using a NN model, on a reverse Duffing system put into the observable
# canonical form with flatness (flat output = x1)

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
    train = True
    # torch.manual_seed(0)
    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = 0

    # General params
    config = Config(system='Continuous/Reverse_Duffing/Discrete_model/'
                           'Back_physical/MLP2_noisy_inputs',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=100,  # TODO
                    t0_span=0,
                    tf_span=6,
                    t0=0,
                    tf=6,
                    init_state_obs_method='KKLu',
                    setD_method='direct',
                    init_state_obs_T=10,
                    NODE_file='../Figures/Continuous/Reverse_Duffing/Discrete_model/'
                              'Back_physical/MLP1_noisy_inputs/'
                              'Test2_100samples_noise0.001_MLP1_Adam0.05/Learn_NODE.pkl',
                    true_meas_noise_var=1e-3,
                    process_noise_var=0,
                    simu_solver='dopri5',  # good solver needed for z
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-6, 'atol': 1e-8},
                    trainer_options={'max_epochs': 500, 'gpus': gpus},
                    optim_method=torch.optim.Adam,
                    optim_lr=7e-2,
                    optim_minibatch_size=100,
                    optim_shuffle=False,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.8, 'patience': 20,
                        'threshold': 0.05, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.005, patience=50,
                        verbose=False, mode='min'),
                    nb_loops=10,
                    nb_rollouts=10,
                    rollout_length=100,
                    rollout_controller={'random': 3, 'sin_controller_1D': 4,
                                        'null_controller': 3},
                    max_rollout_value=5.,
                    verbose=False,
                    monitor_experiment=True,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    plot_output=True)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))

    # Train whole model
    if train:
        # System params
        if 'Continuous/Reverse_Duffing' in config.system:
            config.update(dict(
                discrete=False,
                gamma=0.11,
                omega=1.2,
                dt=config.dt,
                dt_before_subsampling=0.001,
                dynamics_function=Reverse_Duffing,
                controller=sin_controller_1D,
                init_control=torch.tensor([[0.]]),
                observe_data=dim1_observe_data,
                observe_data_x=dim1_observe_data,
                # observe_data=lambda x: x,  # TODO
                # observe_data_x=lambda x: x,
                observer_function=MSM_justvelocity_observer_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10.,
                                                 'k1': 1.,
                                                 'k2': 1.}}))
            dynamics = config.dynamics_function(config.cuda_device, config)
            config.update(dict(
                dynamics=dynamics,
                dynamics_x=dynamics.reverse_duffing_dynamics_x,
                true_dynamics=dynamics.reverse_duffing_dynamics_x))

            # Set initial states of x, xi
            init_state_x = torch.tensor([[0., 1.]])  # Transfo not inv in (x1,0)
            # # Random initial state
            # xmin = torch.tensor([-1., -1.])
            # xmax = torch.tensor([1., 1.])
            # init_state_x = reshape_pt1(torch.rand((1, 2)) * (xmax - xmin) + xmin)
            init_state = config.dynamics.reverse_duffing_xtoz(
                init_state_x.clone())
            init_state_estim = init_state_x  # init state model after
            config.update(dict(init_state_x=init_state_x,
                               init_state=init_state,
                               init_state_estim=init_state_estim,
                               n=init_state_x.shape[1]))

            # Create NN submodel of dynamics, then pack into Learn_NODE
            submodel = MLP2(n_in=config.n + config.init_control.shape[1],
                            n_h1=10, n_h2=10, n_out=config.n,
                            activation=nn.Tanh)  # TODO
            n_param, param = get_parameters(submodel, verbose=True)
            observer = config.observer_function(config.cuda_device, config)
            if config.get('gamma') == 0:
                no_control = True
                nu_submodel = 0
                if config.init_state_obs_method == 'KKLu':
                    logging.warning('KKLu without control: switching to '
                                    'KKL_u0T to ignore the terms in u')
                    config.init_state_obs_method = 'KKL_u0T'
            else:
                no_control = False
                nu_submodel = config.init_control.shape[1]
            config.update(dict(
                n_param=n_param, nu=config.init_control.shape[1],
                reg_coef=1., observer=observer,
                observer_prior_mean=dynamics.reverse_duffing_dynamics_z_justvelocity_true,
                no_control=no_control,
                constrain_u=[-config.get('gamma'),
                             config.get('gamma')],
                constrain_x=[],
                grid_inf=-2,
                grid_sup=2))
        else:
            raise Exception('Unknown system')

        # Simulate system in x
        xtraj_true, utraj, t_utraj = \
            simulate_dynamics(t_eval=config.t_eval, t0=config.t0, dt=config.dt,
                              init_control=config.init_control,
                              init_state=config.init_state_x,
                              dynamics=config.dynamics_x,
                              controller=config.controller,
                              process_noise_var=config.process_noise_var,
                              method=config.simu_solver,
                              dyn_config=config,
                              discrete=config.discrete,
                              verbose=config.verbose)
        y_observed_true = config.observe_data(xtraj_true)
        if config.true_meas_noise_var != 0:
            y_observed_true += torch.normal(0, np.sqrt(
                config.true_meas_noise_var), size=y_observed_true.shape)

        # Simulate corresponding system in z with HGO to select right solution
        # (not Lipschitz form so several solutions possible)
        # TODO use HGO from true x0=eq to select traj, but numerically not
        #  perfect. We would not have this problem for strongly observable
        #  system, so choose different x0 and corresponding z0 and focus on that
        xtraj, utraj, t_utraj = \
            simulate_dynamics(t_eval=config.t_eval, t0=config.t0, dt=config.dt,
                              init_control=config.init_control,
                              init_state=config.init_state,
                              dynamics=config.dynamics,
                              controller=config.controller,
                              process_noise_var=config.process_noise_var,
                              method=config.simu_solver,
                              dyn_config=config,
                              discrete=config.discrete,
                              verbose=config.verbose)  # only for verification
        y_observed = config.observe_data(xtraj)
        if config.true_meas_noise_var != 0:
            y_observed += torch.normal(0, np.sqrt(
                config.true_meas_noise_var), size=y_observed.shape)
        # y_observed, t_y, xtraj = \
        #     simulate_estimations(system=config.system,
        #                          observe_data=config.observe_data,
        #                          t_eval=config.t_eval, t0=config.t0,
        #                          tf=config.tf, dt=config.dt,
        #                          meas_noise_var=config.true_meas_noise_var,
        #                          init_control=config.init_control,
        #                          init_state_estim=config.init_state_estim,
        #                          controller=config.controller,
        #                          observer=config.observer,
        #                          method=config.simu_solver,
        #                          dyn_config=config, xtraj=xtraj_true,
        #                          GP=config.observer_prior_mean,
        #                          discrete=config.discrete,
        #                          verbose=config.verbose)
        if config.verbose:
            # Plot trajectories
            for i in range(xtraj.shape[1]):
                plt.plot(xtraj[:, i], label='z_' + str(i))
                plt.title('z derived ' + str(i) + ' times')
                plt.xlabel('t')
                plt.ylabel('z_' + str(i))
                plt.legend()
                plt.show()
                plt.clf()
                plt.close('all')
            plt.plot(utraj, label='u')
            plt.title('Control trajectory')
            plt.xlabel('t')
            plt.ylabel('u')
            plt.legend()
            plt.show()
            plt.clf()
            plt.close('all')

        if config.verbose:
            # Check invertibility of transformation
            plt.plot(xtraj_true[:, 0], label='x')
            plt.plot(config.dynamics.reverse_duffing_ztox(xtraj)[:, 0],
                     label='T*(''z)')
            plt.plot(config.dynamics.reverse_duffing_ztox(
                config.dynamics.reverse_duffing_xtoz(xtraj_true))[:, 0],
                     label='T*(T(x))')
            plt.legend()
            plt.show()
            plt.clf()
            plt.close('all')
            plt.plot(xtraj_true[:, 1], label='x')
            plt.plot(config.dynamics.reverse_duffing_ztox(xtraj)[:, 1],
                     label='T*(z)')
            plt.plot(config.dynamics.reverse_duffing_ztox(
                config.dynamics.reverse_duffing_xtoz(xtraj_true))[:, 1],
                     label='T*(T(x))')
            plt.legend()
            plt.show()
            plt.clf()
            plt.close('all')

        # Recognition model to estimate x0 jointly with the dynamics
        # First define some configs for the inputs of the recognition model
        if 'KKL_u0T' in config.init_state_obs_method:
            dz = y_observed_true.shape[1] * (config.n + 1)
            W0 = 2 * np.pi * 1
            zO, pO, kO = scipy.signal.bessel(dz, W0, analog=True, output='zpk')
            D, F = setD(pO, dz, X_train.shape[-1], config.setD_method)
            F = torch.ones(dz, y_observed_true.shape[1])
            z0 = torch.zeros(1, dz)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'Db': D, 'Fb': F, 'Bessel_W0': W0}
            KKL = KKL(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))
        elif 'KKLu' in config.init_state_obs_method:
            dw = 3  # to generate sinus with diff freqs, amplitudes
            dz = (y_observed_true.shape[1] +
                  config.init_control.shape[1]) * (config.n + dw + 1)
            W0 = 2 * np.pi * 1
            zO, pO, kO = scipy.signal.bessel(dz, W0, analog=True, output='zpk')
            D, F = setD(pO, dz,  X_train.shape[-1] +
                        config.init_control.shape[-1], config.setD_method)
            F = torch.ones(dz, y_observed_true.shape[1] +
                           config.init_control.shape[1])
            z0 = torch.zeros(1, dz)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'controller_args': config.controller_args,
                        'Db': -D, 'Fb': -F, 'Bessel_W0': W0}
            KKL = KKLu(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))
            wait = scikit-learn

        # Define the inputs of the recognition model
        init_state_obs = make_init_state_obs(
            y_observed_true, utraj, init_state_x, config.t_eval, config)

        # Define the actual recognition model (same for all init)
        init_state_model = MLP2(n_in=torch.numel(init_state_obs),
                                n_h1=10, n_h2=10, n_out=config.n,
                                activation=nn.Tanh)  # TODO
        xtraj_true, y_observed_true, utraj, config.t_eval = \
            update_config_init_state_obs(
                init_state_obs, init_state_model, xtraj_true,
                y_observed_true, utraj, config.t_eval, config)

        # Create Learn_NODE object
        NODE = Learn_NODE(y_observed_true, utraj, submodel, config,
                          sensitivity=config.sensitivity)
        # Train
        # To see logger in tensorboard, type in terminal then click on weblink:
        # tensorboard --logdir=../Figures/Continuous/Reverse_Duffing/Discrete_model/Back_physical/MLP1_noisy_inputs/ --port=8080
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
                    log_every_n_steps=1, check_val_every_n_epoch=10)
            else:
                trainer = pl.Trainer(
                    callbacks=[checkpoint_callback], **config.trainer_options,
                    logger=logger, log_every_n_steps=1,
                    check_val_every_n_epoch=10)
            trainer.fit(NODE)
        # Save and update estim x0
        NODE.save_model(checkpoint_path=checkpoint_callback.best_model_path)
        # Plot additional results
        xtraj_estim = NODE.NODE_model.forward_traj(
            NODE.init_state_estim, config.controller, config.t0,
            config.t_eval, config.init_control)
        for i in range(xtraj_estim.shape[1]):
            name = 'xtraj_estim' + str(i) + '.pdf'
            plt.plot(xtraj_true[:, i], label='True')
            plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
            plt.title('State trajectory')
            plt.legend()
            plt.savefig(os.path.join(NODE.results_folder, name),
                        bbox_inches='tight')
            plt.clf()
            plt.close('all')
    else:
        NODE = pkl.load(open(config.NODE_file, 'rb'))
        NODE.results_folder = config.NODE_file.rsplit('/', 1)[0]
        config = NODE.config

    # # Evaluate model
    # utraj = config.controller(config.t_eval, config, config.t0,
    #                           config.init_control)
    # x0 = torch.tensor([[0.5, 1.5]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')
    # x0 = torch.tensor([[1., 3.5]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')
    # x0 = torch.tensor([[-0.2, -0.8]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
