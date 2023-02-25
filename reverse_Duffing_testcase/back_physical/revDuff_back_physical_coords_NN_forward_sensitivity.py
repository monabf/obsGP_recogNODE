import logging
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import os
import sys

import numpy as np
import seaborn as sb
import dill as pkl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

sys.path.append('.')

from simulation.controllers import sin_controller_1D
from simulation.dynamics import dynamics_traj
from simulation.dynamics_functions import Reverse_Duffing
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data, \
    MSM_justvelocity_observer_highgain_GP
from revDuff_NN_models import MLP1
from NN_for_ODEs.learn_neural_ODE import Learn_NODE
from utils.utils import reshape_pt1, kronecker, \
    reshape_pt1_tonormal, start_log, stop_log, save_log
from utils.pytorch_utils import EarlyStopping, get_parameters, \
    StandardScaler
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


if __name__ == '__main__':
    start_log()
    train = True

    # General params
    config = Config(system='Continuous/Reverse_Duffing/Discrete_model/'
                           'Back_physical/MLP1_forward_sensitivity_noisy_inputs',
                    forward_sensitivity=True,
                    nb_samples=100,  # TODO
                    t0_span=0,
                    tf_span=6,
                    t0=0,
                    tf=6,
                    ODE_NN_file='../Figures/Continuous/Reverse_Duffing/Discrete_model/'
                                'Back_physical/MLP1_noisy_inputs/x0=01_noHGO_noU_noreg_tiny/'
                                'Good1_100samples_noise0.001_MLP1_Adam0.05/NN_for_ODEs.pkl',
                    hyperparam_optim=None,
                    true_meas_noise_var=1e-3,
                    process_noise_var=0,
                    meas_noise_var=0.01,
                    simu_solver='dopri8',  # good solver needed for z
                    # solver_options={'rtol': 1e-6, 'atol': 1e-8},
                    nb_loops=10,
                    nb_rollouts=10,
                    rollout_length=200,
                    rollout_controller={'random': 3, 'sin_controller_1D': 4,
                                        'null_controller': 3},
                    max_rollout_value=10.,
                    sliding_window_size=5000,
                    sliding_window_size_each=None,
                    verbose=False,
                    monitor_experiment=True,
                    multitask_GP=False,
                    sparse=None,
                    memory_saving=False,
                    restart_on_loop=False,
                    optim_method=torch.optim.Adam,
                    optim_training_iter=300,
                    optim_lr=5e-2,
                    # optim_options={'line_search_fn': 'strong_wolfe'},
                    # optim_options={'amsgrad': True},
                    optim_minibatch_size=500,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.8, 'patience': 20,
                        'threshold': 0.05},
                    optim_stopper=EarlyStopping(patience=25, threshold=0.01),
                    batch_adaptive_gain=None,
                    prior_mean=None,
                    prior_mean_deriv=None)
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
                gamma=0.,  # TODO
                omega=1.2,
                dt=config.dt,
                dt_before_subsampling=0.001,
                dynamics_function=Reverse_Duffing,
                controller=sin_controller_1D,
                init_control=torch.tensor([[0.]]),
                observe_data=dim1_observe_data,
                observe_data_x=dim1_observe_data,
                observer_function=MSM_justvelocity_observer_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10.,
                                                 'k1': 1.,
                                                 'k2': 1.}},
                optim_stopper_patience=config.optim_stopper.patience,
                optim_stopper_threshold=config.optim_stopper.threshold))
            dynamics = config.dynamics_function(config.cuda_device, config)
            config.update(dict(dynamics=dynamics,
                               dynamics_x=dynamics.reverse_duffing_dynamics_x))
            init_state_x = torch.tensor(
                [[0., 1.]])  # Transfo not inv in (x1,0)!
            # # Random initial state
            # xmin = torch.tensor([-1., -1.])
            # xmax = torch.tensor([1., 1.])
            # init_state_x = reshape_pt1(torch.rand((1, 2)) * (xmax - xmin) + xmin)
            init_state = config.dynamics.reverse_duffing_xtoz(
                init_state_x.clone())
            init_state_estim = init_state_x  # TODO
            config.update(dict(init_state_x=init_state_x,
                               init_state=init_state,
                               init_state_estim=init_state_estim,
                               n=init_state_estim.shape[1]))
            # Create NN model of dynamics
            submodel = MLP1(n_in=config.n, n_h=10, n_out=config.n,
                         activation=nn.Tanh)
            n_param, param = get_parameters(submodel, verbose=True)
            observer = config.observer_function(config.cuda_device, config)
            config.update(dict(
                model=submodel, n_param=n_param, reg_coef=1., observer=observer,
                observer_prior_mean=dynamics.reverse_duffing_dynamics_z_justvelocity_true))
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
            plt.plot(utraj, label='u')
            plt.title('Control trajectory')
            plt.xlabel('t')
            plt.ylabel('u')
            plt.legend()
            plt.show()

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
            plt.plot(xtraj_true[:, 1], label='x')
            plt.plot(config.dynamics.reverse_duffing_ztox(xtraj)[:, 1],
                     label='T*(z)')
            plt.plot(config.dynamics.reverse_duffing_ztox(
                config.dynamics.reverse_duffing_xtoz(xtraj_true))[:, 1],
                     label='T*(T(x))')
            plt.legend()
            plt.show()

        # Create NN_for_ODEs object
        ODE_NN = Learn_ODE_NN(xtraj_true, utraj, submodel, config,
                              forward_sensitivity=config.forward_sensitivity)
        # Train
        if config.forward_sensitivity:
            traj_estim = ODE_NN.train_forward_sensitivity()
        # Train
        traj_estim = ODE_NN.train_with_sensitivity(None, None)  # TODO modular
        # Save
        ODE_NN.save_model()

        # Show results
        xtraj_estim = traj_estim[:, :config.n]
        lambdatraj_estim = traj_estim[:, config.n:]
        lambdatraj_estim = lambdatraj_estim.reshape(
            -1, config.n_param, config.n).permute(0, 2, 1)
        y_pred = config.observe_data_x(xtraj_estim)
        # logging.info('Initial parameter: ' + str(config.init_param))
        # logging.info('Final parameter: ' + str(config.param))
        # Plots
        name = 'Loss.pdf'
        plt.plot(ODE_NN.losses, '+-', label='loss')
        plt.title('Loss over time')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(ODE_NN.results_folder, name),
                    bbox_inches='tight')
        plt.show()
        for i in range(xtraj_estim.shape[1]):
            name = 'xtraj_estim' + str(i) + '.pdf'
            plt.plot(xtraj_true[:, i], label='True')
            plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
            plt.title('State trajectory')
            plt.legend()
            plt.savefig(os.path.join(ODE_NN.results_folder, name),
                        bbox_inches='tight')
            plt.show()
        for i in range(y_pred.shape[1]):
            name = 'y_pred' + str(i) + '.pdf'
            plt.plot(y_observed[:, i], label='True')
            plt.plot(y_pred.detach()[:, i], label='Estimated')
            plt.title('Output')
            plt.legend()
            plt.savefig(os.path.join(ODE_NN.results_folder, name),
                        bbox_inches='tight')
            plt.show()
    else:
        ODE_NN = pkl.load(open(config.ODE_NN_file, 'rb'))
        config = ODE_NN.config


    # Evaluate model
    dyns = lambda t, x, u, t0, init_control, process_noise_var, kwargs: \
        ODE_NN.model(x)
    x0 = torch.tensor([[0.5, 1.5]])
    xtraj_estim = dynamics_traj(x0=x0,
                                u=config.controller,
                                t0=config.t0, dt=config.dt,
                                init_control=config.init_control,
                                discrete=config.discrete,
                                version=dyns,
                                meas_noise_var=0,
                                process_noise_var=0,
                                method=config.solver_optim_method,
                                t_eval=config.t_eval,
                                kwargs=config)
    xtraj_true = dynamics_traj(x0=x0,
                               u=config.controller,
                               t0=config.t0, dt=config.dt,
                               init_control=config.init_control,
                               discrete=config.discrete,
                               version=config.dynamics_x,
                               meas_noise_var=0,
                               process_noise_var=0,
                               method=config.solver_optim_method,
                               t_eval=config.t_eval,
                               kwargs=config)
    for i in range(xtraj_estim.shape[1]):
        name = 'xtraj_estim' + str(i) + '.pdf'
        plt.plot(xtraj_true[:, i], label='True')
        plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
        plt.title('State trajectory')
        plt.legend()
        plt.show()
    x0 = torch.tensor([[-0.1, -0.8]])
    xtraj_estim = dynamics_traj(x0=x0,
                                u=config.controller,
                                t0=config.t0, dt=config.dt,
                                init_control=config.init_control,
                                discrete=config.discrete,
                                version=dyns,
                                meas_noise_var=0,
                                process_noise_var=0,
                                method=config.solver_optim_method,
                                t_eval=config.t_eval,
                                kwargs=config)
    xtraj_true = dynamics_traj(x0=x0,
                               u=config.controller,
                               t0=config.t0, dt=config.dt,
                               init_control=config.init_control,
                               discrete=config.discrete,
                               version=config.dynamics_x,
                               meas_noise_var=0,
                               process_noise_var=0,
                               method=config.solver_optim_method,
                               t_eval=config.t_eval,
                               kwargs=config)
    for i in range(xtraj_estim.shape[1]):
        name = 'xtraj_estim' + str(i) + '.pdf'
        plt.plot(xtraj_true[:, i], label='True')
        plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
        plt.title('State trajectory')
        plt.legend()
        plt.show()
    x0 = torch.tensor([[2., 0.3]])
    xtraj_estim = dynamics_traj(x0=x0,
                                u=config.controller,
                                t0=config.t0, dt=config.dt,
                                init_control=config.init_control,
                                discrete=config.discrete,
                                version=dyns,
                                meas_noise_var=0,
                                process_noise_var=0,
                                method=config.solver_optim_method,
                                t_eval=config.t_eval,
                                kwargs=config)
    xtraj_true = dynamics_traj(x0=x0,
                               u=config.controller,
                               t0=config.t0, dt=config.dt,
                               init_control=config.init_control,
                               discrete=config.discrete,
                               version=config.dynamics_x,
                               meas_noise_var=0,
                               process_noise_var=0,
                               method=config.solver_optim_method,
                               t_eval=config.t_eval,
                               kwargs=config)
    for i in range(xtraj_estim.shape[1]):
        name = 'xtraj_estim' + str(i) + '.pdf'
        plt.plot(xtraj_true[:, i], label='True')
        plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
        plt.title('State trajectory')
        plt.legend()
        plt.show()

    stop_log()
