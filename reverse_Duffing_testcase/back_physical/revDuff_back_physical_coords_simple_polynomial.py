import logging
import os
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import sys
import time

import numpy as np
import seaborn as sb
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

sys.path.append('.')

from simulation.controllers import sin_controller_1D
from simulation.dynamics import dynamics_traj
from simulation.dynamics_functions import Reverse_Duffing
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data, \
    MSM_justvelocity_observer_highgain_GP
from utils.utils import reshape_pt1, start_log, stop_log, save_log, kronecker, \
    reshape_pt1_tonormal
from utils.pytorch_utils import EarlyStopping
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
# using a simple polynomial dynamics model, on a reverse Duffing system put
# into the observable canonical form with flatness (flat output = x1)

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


# Dynamics function to simulate a trajectory (x, lambd)
def dynamics_ext(t, x, u, t0, init_control, process_noise_var,
                 kwargs, impose_init_control=False, verbose=False):
    start = time.time()
    x = reshape_pt1(x)
    n = kwargs.get('init_state_estim').shape[1]
    nu = kwargs.get('init_control').shape[1]
    lamb = reshape_pt1(x[:, n:])
    x = reshape_pt1(x[:, :n])
    u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
    param = kwargs.get('param').reshape(n, -1).t()  # matrix param
    param_u = param[-nu:]  # vector param for u
    param_x = param[:-nu]  # matrix param for x
    phi = torch.tensor([
        x[:, 0], torch.pow(x[:, 0], 2), torch.pow(x[:, 0], 3),
        x[:, 1], torch.pow(x[:, 1], 2), torch.pow(x[:, 1], 3)])
    dphidx = torch.tensor([
        [1, 2 * x[:, 0], 3 * torch.pow(x[:, 0], 2), 0, 0, 0],
        [0, 0, 0, 1, 2 * x[:, 1], 3 * torch.pow(x[:, 1], 2)]]).t()
    # xdot = reshape_pt1(torch.matmul(param.t(), phi))
    xdot = reshape_pt1(torch.matmul(param_x.t(), phi)) + (
            param_u.t() * u).t()
    dfdx = torch.matmul(param_x.t(), dphidx)
    dfdparam = torch.cat((kronecker(torch.eye(n), reshape_pt1(
        phi)), reshape_pt1(u) * torch.eye(n)), dim=1)
    param = param.t().flatten()  # back flattened param
    lambdot = reshape_pt1(torch.matmul(kronecker(
        torch.eye(len(param)), dfdx), reshape_pt1_tonormal(lamb)) +
                          dfdparam.t().flatten())
    end = time.time()
    if verbose:
        print(f'simu in {end - start} s')
    return torch.cat((xdot, lambdot), dim=1)


if __name__ == '__main__':
    start_log()

    # General params
    config = Config(system='Continuous/Reverse_Duffing/Discrete_model/'
                           'Back_physical/Simple_poly3_xdot_noisy_inputs',
                    nb_samples=100,
                    t0_span=0,
                    tf_span=6,
                    t0=0,
                    tf=6,
                    hyperparam_optim=None,
                    true_meas_noise_var=1e-3,
                    process_noise_var=0,
                    meas_noise_var=0.01,
                    simu_solver='dopri8',  # good solver needed for z
                    solver_options={'rtol': 1e-6, 'atol': 1e-8},
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
                    optim_lr=5e-3,
                    # optim_options={'line_search_fn': 'strong_wolfe'},
                    # optim_options={'amsgrad': True},
                    optim_minibatch_size=500,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.8, 'patience': 10,
                        'threshold': 0.005},
                    optim_stopper=EarlyStopping(patience=25, threshold=0.8),
                    batch_adaptive_gain=None,
                    prior_mean=None,
                    prior_mean_deriv=None)
    t_eval = torch.linspace(config.t0, config.tf, config.nb_samples)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))

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
        # init_state_x = torch.tensor([[0.5, 0.5]])  # Transfo not inv in (x1,0)!
        # Random initial state
        xmin = torch.tensor([-1., -1.])
        xmax = torch.tensor([1., 1.])
        init_state_x = reshape_pt1(torch.rand((1, 2)) * (xmax - xmin) + xmin)
        init_state = config.dynamics.reverse_duffing_xtoz(init_state_x.clone())
        # init_state_estim = torch.cat((
        #     config.observe_data(reshape_pt1(init_state)),
        #     reshape_pt1(torch.tensor([[0.] * (init_state.shape[1] - 1)]))),
        #     dim=1)  # TODO
        init_state_estim = init_state_x
        config.update(dict(init_state_x=init_state_x,
                           init_state=init_state,
                           init_state_estim=init_state_estim))
        # init_param = 0.001 * torch.rand((6, config.init_state_estim.shape[1]))
        init_param = torch.tensor([[0., 0., 0., 0., 0., 0.9],  # init with ERA?
                                   [-0.9, 0., 0., 0., 0., 0.]]).t()  # TODO
        init_param = init_param + 0.001 * torch.rand(init_param.shape)
        init_param_u = torch.tensor([[0., 0.9]])
        init_param_u = init_param_u + 0.001 * torch.rand(init_param_u.shape)
        init_param = torch.cat((init_param, init_param_u), dim=0)
        observer = config.observer_function(config.cuda_device, config)
        config.update(dict(
            init_param=init_param, reg_coef=1., observer=observer,
            observer_prior_mean=dynamics.reverse_duffing_dynamics_z_justvelocity_true))
    else:
        raise Exception('Unknown system')

    # Put all tensor data on cuda
    if torch.cuda.is_available():
        for key, val in config.items():
            if torch.is_tensor(val):
                print(key)
                val = val.cuda()

    # Simulate system in x
    xtraj_true, utraj, t_utraj = \
        simulate_dynamics(t_eval=t_eval, t0=config.t0, dt=config.dt,
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
        simulate_dynamics(t_eval=t_eval, t0=config.t0, dt=config.dt,
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
    #                          t_eval=t_eval, t0=config.t0, tf=config.tf,
    #                          dt=config.dt,
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

    # Prepare parameter and optimizer
    param = reshape_pt1(
        config.init_param.t().flatten()).clone().detach().requires_grad_(True)
    init_ext = torch.cat((config.init_state_estim, torch.zeros((
        1, config.init_state_estim.shape[1] * param.shape[1]))), dim=1)
    # Put all tensor data on cuda
    if torch.cuda.is_available():
        param, init_ext = param.cuda(), init_ext.cuda()
    logging.info('X0: ' + str(init_ext))
    logging.info('param0: ' + str(param))
    config.update(dict(param=param))
    losses = []
    if config.optim_options:
        optim_options = config.optim_options
    else:
        optim_options = {}
    optimizer = config.optim_method([{'params': param}],
                                    config.optim_lr, **optim_options)
    if config.optim_scheduler_options:
        optim_scheduler_options = config.optim_scheduler_options
    else:
        optim_scheduler_options = {}
    scheduler = config.optim_scheduler(optimizer,
                                       **optim_scheduler_options)
    if config.optim_minibatch_size:
        optim_minibatch_size = config.optim_minibatch_size
    else:
        optim_minibatch_size = config.nb_samples

    # Prepare dataset, dataloader and iterators through data
    train_dataset = TensorDataset(torch.arange(len(y_observed)), y_observed)
    train_loader = DataLoader(
        train_dataset, batch_size=optim_minibatch_size, shuffle=True)
    epochs_iter = tqdm.tqdm(range(config.optim_training_iter), desc="Epoch",
                            leave=True)
    for k in epochs_iter:
        # Forward pass: in each epoch and each minibatch, solve training data
        # (x_estim, lambda_estim) for current param, then optimize loss to get
        # new param. Minibatches go over fixed data = (idx_yobs, yobs)
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch",
                                   leave=False)  # nb depends size
        for idx_batch, y_batch in minibatch_iter:
            y_batch = reshape_pt1(y_batch)
            param_before = param.clone()

            # Simulate x, lambda  # TODO each minibatch = new simu!
            traj_estim = dynamics_traj(x0=init_ext,
                                       u=config.controller,
                                       t0=config.t0, dt=config.dt,
                                       init_control=config.init_control,
                                       discrete=config.discrete,
                                       version=dynamics_ext,
                                       meas_noise_var=0,
                                       process_noise_var=0,
                                       method=config.solver_optim_method,
                                       t_eval=t_eval,
                                       kwargs=config)
            # Organize simulation results in (x, lambda) and keep only minibatch
            n = config.get('init_state_estim').shape[1]
            xtraj_estim = traj_estim[idx_batch, :n]
            lambdatraj_estim = traj_estim[idx_batch, n:]
            lambdatraj_estim = lambdatraj_estim.reshape(
                -1, param.shape[1], n).permute(0, 2, 1)
            y_pred = config.observe_data_x(xtraj_estim)

            # Compute loss, its gradient, step of optimizer and optimize param
            if 'LBFGS' in optimizer.__class__.__name__:
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    loss = 1. / 2 * torch.sum(torch.square(y_pred - y_batch),
                                              dim=0)
                    # regularization
                    loss += config.reg_coef * 1. / 2 * torch.sum(torch.square(
                        param - config.init_param.t().flatten()))
                    losses.append(loss.item())
                    minibatch_iter.set_postfix(loss=loss.item())
                    C = torch.zeros(xtraj_estim.shape[1], )
                    C[0] = 1
                    dloss = torch.sum((y_pred - y_batch) * torch.matmul(
                        C, lambdatraj_estim), dim=0)
                    # regularization
                    dloss += config.reg_coef * reshape_pt1_tonormal(
                        param - config.init_param.t().flatten())
                    if loss.requires_grad:
                        # manually set gradient
                        param.backward(reshape_pt1(dloss))
                    return loss


                # Take a step of gradient descent
                optimizer.step(closure)
                config.update(dict(param=param))
                loss = losses[-1]
            else:
                loss = 1. / 2 * torch.sum(torch.square(y_pred - y_batch), dim=0)
                # regularization
                loss += config.reg_coef * 1. / 2 * torch.sum(torch.square(
                    param - config.init_param.t().flatten()))
                losses.append(loss.item())
                minibatch_iter.set_postfix(loss=loss.item())
                C = torch.zeros(xtraj_estim.shape[1], )
                C[0] = 1
                dloss = torch.sum((y_pred - y_batch) * torch.matmul(
                    C, lambdatraj_estim), dim=0)
                # regularization
                dloss += config.reg_coef * reshape_pt1_tonormal(
                    param - config.init_param.t().flatten())

                # Take a step of gradient descent
                with torch.no_grad():
                    param.backward(reshape_pt1(dloss))  # manually set gradient
                    optimizer.step()
                    config.update(dict(param=param))
                    # Manually zero the gradients after updating weights
                    optimizer.zero_grad()

        # n = config.init_state_estim.shape[1]
        # xtraj_estim = traj_estim[:, :n]
        # lambdatraj_estim = traj_estim[:, n:]
        # lambdatraj_estim = lambdatraj_estim.reshape(
        #     -1, param.shape[1], n).permute(0, 2, 1)
        # for i in range(xtraj_estim.shape[1]):  # TODO
        #     name = 'xtraj_estim' + str(i) + '.pdf'
        #     plt.plot(xtraj_true[:, i], label='True')
        #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
        #     plt.title('State trajectory')
        #     plt.legend()
        #     plt.show()

        # Update the learning rate and the early stopping at each epoch
        epochs_iter.set_postfix(loss=loss.item())
        scheduler.step(loss)
        config.optim_stopper(loss)
        print(k, loss, config.param, param_before, scheduler.state_dict(),
              config.optim_stopper.best_loss)
        if config.optim_stopper.early_stop:
            # Stop early
            break

    # Show and save results
    n = config.init_state_estim.shape[1]
    xtraj_estim = traj_estim[:, :n]
    lambdatraj_estim = traj_estim[:, n:]
    lambdatraj_estim = lambdatraj_estim.reshape(
        -1, param.shape[1], n).permute(0, 2, 1)
    y_pred = config.observe_data_x(xtraj_estim)
    config.param = config.param.reshape(n, -1).t()
    params = str(np.random.uniform()) + '_noise' + str(
        config.true_meas_noise_var) + '_' + str(
        config.optim_method.__name__) + '_' + str(config.optim_lr)
    results_folder = os.path.join('../Figures', str(config.system), params)
    os.makedirs(results_folder, exist_ok=False)
    specs_file = os.path.join(results_folder, 'Specifications.txt')
    with open(specs_file, 'w') as f:
        for key, val in config.params.items():
            print(key, ': ', val, file=f)
    # Log
    os.rename('../Figures/Logs/' + 'log' + str(sys.argv[1]) + '.log',
              os.path.join(results_folder, 'log' + str(sys.argv[1]) + '.log'))
    save_log(results_folder)
    logging.info('Initial parameter: ' + str(config.init_param))
    logging.info('Final parameter: ' + str(config.param))
    # Plots
    name = 'Loss.pdf'
    plt.plot(losses, '+-', label='loss')
    plt.title('Loss over time')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
    plt.show()
    for i in range(xtraj_estim.shape[1]):
        name = 'xtraj_estim' + str(i) + '.pdf'
        plt.plot(xtraj_true[:, i], label='True')
        plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
        plt.title('State trajectory')
        plt.legend()
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.show()
    for i in range(y_pred.shape[1]):
        name = 'y_pred' + str(i) + '.pdf'
        plt.plot(y_observed[:, i], label='True')
        plt.plot(y_pred.detach()[:, i], label='Estimated')
        plt.title('Output')
        plt.legend()
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.show()

    stop_log()
