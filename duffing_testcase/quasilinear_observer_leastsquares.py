import logging
import os
import sys
from copy import deepcopy

import dill as pkl
import gpytorch
import torch
import GPy
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import linear_model
from torchdiffeq import odeint

sys.path.append('.')

from simulation.controllers import sin_controller_1D
from simulation.dynamics_functions import Duffing
from simulation.observers import dim1_observe_data
from simulation.observer_functions import MSM_observer_Michelangelo_GP, \
    MSM_justvelocity_observer_highgain_GP
from simulation.prior_means import Duffing_prior
from simulation.simulation_functions import simulate_dynamics, \
    simulate_estimations, form_GP_data
from utils import reshape_pt1, reshape_dim1, RMS, reshape_pt1_tonormal, rk4, \
    euler, Config

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

# Script for Duffing HGO test case, but with parametric model and least
# squares estimation instead of the GP
# Old script so self contained, though a bit cluttered/ugly

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


def start_log():
    logging.INFO
    logging.FileHandler("{0}/{1}.log".format(
        '../Figures/Logs', 'log' + str(sys.argv[1])))
    logging.StreamHandler(sys.stdout)


def stop_log():
    logging._handlers.clear()
    logging.shutdown()


# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# High gain extended observer for the continuous time Duffing equation
# Using current LS estimation for xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
def duffing_observer_Michelangelo_LS(t, xhat, u, y, t0, init_control, LS_deriv,
                                     impose_init_control, kwargs):
    x = reshape_pt1(xhat)
    xhat = reshape_pt1(x[:, :-1])
    xi = reshape_pt1(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    Gamma1 = torch.tensor([k1 * g, k2 * g ** 2])
    Gamma2 = torch.tensor([k3 * g ** 3])
    if LS_deriv:
        mean_deriv = LS_deriv(reshape_pt1(xhat), reshape_pt1(u),
                              kwargs.get('prior_kwargs'))
    else:
        mean_deriv = torch.zeros_like(xhat)
    if kwargs.get('saturation') is not None:
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean_deriv = torch.clip(torch.as_tensor(mean_deriv),
                                min=torch.as_tensor(a_min),
                                max=torch.as_tensor(a_max))
    A = torch.tensor([[0., 1], [0, 0.]])
    B = torch.tensor([[0.], [1.]])
    ABmult = torch.matmul(A, reshape_pt1_tonormal(xhat)) + \
             torch.matmul(B, reshape_pt1_tonormal(xi + u))
    DfA = torch.matmul(reshape_pt1_tonormal(mean_deriv),
                       reshape_pt1_tonormal(ABmult))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    LC2 = reshape_pt1(Gamma2 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult) + LC1
    xidot = reshape_pt1(DfA) + LC2

    # Also check eigenvalues of M for stability without high gain
    AB = torch.cat((A, B), dim=1)
    ABO = torch.cat((AB, torch.zeros_like(reshape_pt1(AB[0]))), dim=0)
    K = torch.tensor([[k1, k2, k3]])
    C = torch.zeros_like(x)
    C[0, 0] = 1.
    M = ABO - torch.matmul(K.T, C)
    eigvals = torch.linalg.eigvals(M)
    for x in eigvals:
        if torch.linalg.norm(torch.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif torch.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return torch.cat((xhatdot, xidot), dim=1)


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current LS estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity
def WDC_justvelocity_observer_highgain_LS(t, xhat, u, y, t0, init_control,
                                          LS, impose_init_control, kwargs):
    x = reshape_pt1(xhat)
    xhat = reshape_pt1(x)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    Gamma1 = torch.tensor([k1 * g, k2 * g ** 2])
    dt = kwargs.get('dt')
    if LS:
        mean = LS(reshape_pt1(xhat), reshape_pt1(u), kwargs.get('prior_kwargs'))
        if not kwargs.get('continuous_model'):
            # discrete model so need to differentiate it in continuous obs
            # mean = (mean - reshape_pt1(xhat[:, 1])) / kwargs.get('dt')
            # TODO better than Euler?
            mean = mean / kwargs.get('dt')  # contains prior mean!
    else:
        mean = torch.zeros_like(reshape_pt1(xhat[:, 1]))
    if kwargs.get('saturation') is not None:
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean = torch.clip(torch.as_tensor(mean), min=torch.as_tensor(a_min),
                          max=torch.as_tensor(a_max))
    A = torch.tensor([[0., 1], [0, 0.]])
    B = torch.tensor([[0.], [1.]])
    ABmult = torch.matmul(A, reshape_pt1_tonormal(xhat)) + \
             torch.matmul(B, reshape_pt1_tonormal(mean + u))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult) + LC1

    # Also check eigenvalues of M for stability without high gain
    K = torch.tensor([[k1, k2]])
    C = torch.zeros_like(xhat)
    C[0, 0] = 1.
    M = A - torch.matmul(reshape_pt1(K).t(), C)
    eigvals = torch.linalg.eigvals(M)
    for x in eigvals:
        if torch.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
        elif torch.linalg.norm(torch.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
    return reshape_pt1(xhatdot)


def plot_traj(predicted_mean_traj, true_mean, time, subfolder, name=None):
    for i in range(predicted_mean_traj.shape[1]):
        if name is None:
            name = f'Rollout_model_predictions{i}.pdf'
        else:
            name = name + f'Rollout_model_predictions{i}.pdf'
        plt.plot(time, true_mean[:, i], 'g', label='True')
        plt.plot(time, predicted_mean_traj[:, i],
                 label='Predicted', c='b', alpha=0.7)
        plt.legend()
        # plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5),
        #            frameon=True)
        plt.xlim(xmin=time[0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x_{}$'.format(i + 1))
        plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        plt.close('all')
    for i in range(predicted_mean_traj.shape[1] - 1):
        if name is None:
            name = f'Rollout_phase_portrait{i}.pdf'
        else:
            name = name + f'Rollout_phase_portrait{i}.pdf'
        plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                 label='True')
        plt.plot(predicted_mean_traj[:, i],
                 predicted_mean_traj[:, i + 1],
                 label='Predicted', c='b', alpha=0.7)
        plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
                    marker='x', s=100, label='Initial state')
        plt.legend()
        # plt.legend(loc='lower left')
        plt.xlabel(r'$x_{}$'.format(i + 1))
        plt.ylabel(r'$x_{}$'.format(i + 2))
        plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        plt.close('all')


def sigma(x):
    # Features used for least squares (big prior knowledge!)
    x = reshape_pt1(x)
    return np.concatenate((reshape_dim1(-x[:, 0]),
                           reshape_dim1(-x[:, 0] ** 3),
                           reshape_dim1(-x[:, 1])), axis=1)


def dsigma_dx(x):
    # Derivative of feature vector
    x = reshape_pt1(x.numpy())
    assert x.shape[0] == 1, 'Can only deal with one point at a time'
    return np.array([[-1, 0.], [-3 * x[:, 0].item() ** 2, 0.], [0., -1]])


if __name__ == '__main__':
    start_log()

    # General params
    config = Config(system='Continuous/Duffing/Discrete_model/'
                           'LS_justvelocity_highgain_observer_noisy_inputs',
                    nb_samples=500,
                    t0_span=0,
                    tf_span=15,
                    t0=0,
                    tf=15,
                    hyperparam_optim='fixed_hyperparameters',
                    true_meas_noise_var=1e-5,  # TODO
                    process_noise_var=0,
                    meas_noise_var=0.1,
                    simu_solver='dopri5',
                    solver_options={'rtol': 1e-3, 'atol': 1e-6},  # scipy def
                    nb_loops=50,
                    nb_rollouts=10,
                    rollout_length=300,
                    rollout_controller={'random': 3, 'sin_controller_1D': 4,
                                        'null_controller': 3},
                    max_rollout_value=5.,
                    sliding_window_size=3000,
                    sliding_window_size_each=None,
                    verbose=False,
                    backwards_after_estim=False,  # TODO
                    monitor_experiment=True,
                    multitask_GP=False,
                    sparse=None,
                    memory_saving=False,
                    restart_on_loop=False,
                    GP_optim_method=torch.optim.Adam,
                    GP_optim_training_iter=100,
                    GP_optim_lr=0.01,
                    batch_adaptive_gain=None,
                    observer_prior_mean=None,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    use_GPy=True,  # TODO
                    plot_output=False)
    t_eval = torch.linspace(config.t0, config.tf, config.nb_samples)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))
    print(config.cuda_device)

    # System params
    if 'Continuous/Duffing' in config.system:
        config.update(dict(
            discrete=False,
            alpha=-1,
            beta=1,
            delta=0.3,
            gamma=0.4,
            omega=1.2,
            dt=config.dt,
            dt_before_subsampling=0.001,
            dynamics_function=Duffing,
            controller=sin_controller_1D,
            init_control=torch.tensor([[0.]])))
        init_state = torch.tensor([[1., 1]])
        init_state_estim = torch.tensor([[0., 0]])
        config.update(dict(init_state=init_state,
                           init_state_estim=init_state_estim))
        if 'LS_Michelangelo' in config.system:
            config.update(dict(
                observer=duffing_observer_Michelangelo_LS,
                prior_kwargs={'alpha': 0, 'beta': 0,
                              'delta': 0, 'gamma': 0.4,
                              'omega': 1.2, 'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 8, 'k1': 5., 'k2': 5,
                                                 'k3': 1}},
                # TODO comparison
                # dyn_kwargs['prior_kwargs']['observer_gains'] = \
                #     {'g': 11., 'k1': 1.52862409, 'k2': 0.9736215, 'k3': 0.24805021}
                saturation=np.array([30]),
                observer_prior_mean=None,
                dyn_GP_prior_mean=None,
                dyn_GP_prior_mean_deriv=None,
                init_state_estim=reshape_pt1(torch.tensor([[0, 0, 0]]))))
        elif 'LS_justvelocity_highgain' in config.system:
            config.update(dict(
                observer=WDC_justvelocity_observer_highgain_LS,
                prior_kwargs={'alpha': 0, 'beta': 0,
                              'delta': 0, 'gamma': 0.4,
                              'omega': 1.2, 'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 8, 'k1': 5., 'k2': 5}},
                # TODO comparison
                # dyn_kwargs['prior_kwargs']['observer_gains'] = \
                #     {'g': 11., 'k1': 1.08827962, 'k2': 0.39478418}
                saturation=np.array([30]),
                observer_prior_mean=None,
                init_state_estim=reshape_pt1(torch.tensor([[0, 0]]))))
            prior = Duffing_prior(config.cuda_device, config.prior_kwargs)
            config.update(dict(
                prior_mean=prior.discrete_justvelocity))
        elif 'No_observer' in config.system:
            config.update(dict(observer_function=None,
                               observer_prior_mean=None,
                               dyn_GP_prior_mean=None,
                               prior_kwargs={'dt': config.dt,
                                             'dt_before_subsampling': 0.001}))
        if config.get('gamma') == 0:
            input_dim = config.init_state.shape[1]
            no_control = True
        else:
            input_dim = config.init_state.shape[1] + \
                        config.init_control.shape[1]
            no_control = False
        config.update(dict(observe_data=dim1_observe_data,
                           constrain_u=[-config.get('gamma'),
                                        config.get('gamma')],
                           constrain_x=[],
                           grid_inf=-2,
                           grid_sup=2,
                           input_dim=input_dim,
                           no_control=no_control,
                           dynamics=config.dynamics_function(
                               config.cuda_device, config)))
        print(config.saturation)
    else:
        raise Exception('Unknown system')

    # Define true model
    if 'LS_justvelocity' in config.system:
        dt = config.get('dt')
        # true_coefs = [config.get('alpha') * dt,
        #               config.get('beta') * dt,
        #               config.get('delta') * dt]
        # Obtained with reglin over true trajectory data (no noise),
        # better than Euler approximation above  # TODO
        # true_coefs = torch.tensor(
        #     [-0.04717311, 0.05200867, -0.99531069])  # tf30
        true_coefs = np.array(
            [-0.02996219, 0.02985621, 0.00972681])  # tf15
    elif 'LS_Michelangelo' in config.system:
        true_coefs = np.array([config.get('alpha'), config.get('beta'),
                               config.get('delta')])
    else:
        raise Exception('True coefs not defined for this system')
    true_coefs_norm = np.linalg.norm(true_coefs)
    init_coef_error = np.linalg.norm(
        np.zeros_like(true_coefs) - true_coefs) / true_coefs_norm
    coef_errors = [init_coef_error]

    # Loop to create data and learn model several times
    for loop in range(config.nb_loops):

        # Adapt parameters on loop start if necessary
        if loop > 0:
            # Update params and initial states after the first pass
            init_state = reshape_pt1(xtraj[-1].clone())
            init_control = reshape_pt1(utraj[-1].clone())
            init_state_estim = reshape_pt1(xtraj_estim[-1].clone())
            tf_before = config.tf
            tf_span = tf_before + (config.tf_span - config.t0_span)
            t0_span = tf_before
            tf = tf_before + (config.tf - config.t0)
            t0 = tf_before
            config.update(dict(t0=t0, tf=tf, t0_span=t0_span,
                               tf_span=tf_span,
                               init_state=init_state,
                               init_control=init_control,
                               init_state_estim=init_state_estim))
            t_eval = torch.linspace(t0, tf, config.nb_samples)

        # Simulate true system
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
                              verbose=config.verbose)

        # Observe data: only position, observer reconstitutes velocity
        # Get observations over t_eval and simulate xhat only over t_eval
        if loop == 0:
            LS_for_observer = config.observer_prior_mean
        else:
            # Update current model for the observer
            current_model = lambda x, u, kwargs: torch.as_tensor(reg.predict(
                sigma(x.numpy())))
            current_model_deriv = lambda x, u, kwargs: torch.as_tensor(
                reshape_pt1(
                    np.dot(reg.coef_, dsigma_dx(x))))
            print(current_model(X[0], U[0], config.get('prior_kwargs')))
            print(
                current_model_deriv(X[0], U[0], config.get('prior_kwargs')),
                X[0])

            if 'LS_Michelangelo' in config.system:
                LS_for_observer = current_model_deriv
            elif 'LS_justvelocity_highgain' in config.system:
                LS_for_observer = current_model
                # current_model_deriv = lambda x, u, kwargs: reshape_pt1(
                #     torch.matmul((reg.coef_ + [0, 0, 1]) / dt, dsigma_dx(x)))
                # print(reg.coef_, reg.coef_ / dt, (reg.coef_ + [0, 0, 1]) / dt)
                # LS_for_observer = current_model_deriv
            else:
                raise Exception('No LS model or not defined')
        y_observed, t_y, xtraj_estim = \
            simulate_estimations(system=config.system,
                                 observe_data=config.observe_data,
                                 t_eval=t_eval, t0=config.t0, tf=config.tf,
                                 dt=config.dt,
                                 meas_noise_var=config.true_meas_noise_var,
                                 init_control=config.init_control,
                                 init_state_estim=config.init_state_estim,
                                 controller=config.controller,
                                 observer=config.observer,
                                 method=config.simu_solver,
                                 dyn_config=config, xtraj=xtraj,
                                 GP=LS_for_observer,
                                 discrete=config.discrete,
                                 verbose=config.verbose)

        # Create data for GP, noiseless or noisy X, noiseless U, noisy Y
        if loop == 0:
            X, U, Y = form_GP_data(system=config.system, xtraj=xtraj,
                                   xtraj_estim=xtraj_estim, utraj=utraj,
                                   # xtraj_estim=xtraj, utraj=utraj,  # TODO noobs
                                   meas_noise_var=config.true_meas_noise_var,
                                   y_observed=y_observed,
                                   derivative_function=config.derivative_function,
                                   model=LS_for_observer)
            if config.prior_mean is not None:
                prior_mean_vector = config.prior_mean(X, U, config.prior_kwargs)
                Y = reshape_pt1(Y - prior_mean_vector)
        else:
            X_old, U_old, Y_old = deepcopy(X), deepcopy(U), deepcopy(Y)
            X, U, Y = form_GP_data(system=config.system, xtraj=xtraj,
                                   xtraj_estim=xtraj_estim, utraj=utraj,
                                   # xtraj_estim=xtraj, utraj=utraj,  # TODO noobs
                                   meas_noise_var=config.true_meas_noise_var,
                                   y_observed=y_observed,
                                   derivative_function=config.derivative_function,
                                   model=LS_for_observer)
            if config.prior_mean is not None:
                prior_mean_vector = config.prior_mean(X, U, config.prior_kwargs)
                Y = reshape_pt1(Y - prior_mean_vector)
            X = torch.cat((X_old, X), dim=0)
            U = torch.cat((U_old, U), dim=0)
            Y = torch.cat((Y_old, Y), dim=0)

        # Fit LS model to updated data
        reg = linear_model.LinearRegression(
            fit_intercept=False)  # No bias term!
        print(X.shape)
        reg.fit(X=sigma(X.numpy()), y=Y.numpy())
        print(reg.coef_)
        # reg.coef_ = np.array(true_coefs)  # TODO true params
        # print(reg.coef_)

        # Save training data
        os.makedirs(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' +
            str(loop), exist_ok=False)
        for j in range(np.min([xtraj.shape[1], xtraj_estim.shape[1]])):
            name = 'xtraj_xtrajestim_' + str(j) + '.pdf'
            plt.plot(xtraj[:, j], label='True state', c='g')
            plt.plot(xtraj_estim[:, j], label='Estimated state', c='orange')
            plt.title('True and estimated position over time')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('x_' + str(j))
            plt.savefig(
                os.path.join(
                    '../Figures/' + config.system +
                    '/Least_squares_sigma/Loop_' + str(loop), name),
                bbox_inches='tight')
            plt.close('all')
            plt.clf()
        name = 'Estimation_error'
        dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
        error = torch.sum(
            torch.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]), dim=1)
        df = pd.DataFrame(error)
        df.to_csv(
            os.path.join(
                '../Figures/' + config.system + '/Least_squares_sigma/Loop_'
                + str(loop), name + '.csv'), header=False)
        plt.plot(error, 'orange', label='True trajectory')
        plt.title('Error plot')
        plt.xlabel('t')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(
            os.path.join(
                '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                    loop),
                name + '.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()

        # Save model results
        coef_error = np.linalg.norm(reg.coef_ - true_coefs) / true_coefs_norm
        specs_file = os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'Specifications.txt')
        with open(specs_file, 'w') as f:
            print(config, file=f)
            print('\n\n\n', file=f)
            print('LS reg score: ', reg.score(X=sigma(X.numpy()), y=Y.numpy()),
                  file=f)
            print('Data shape: ', X.shape, Y.shape, file=f)
            print('LS reg coef: ', reg.coef_, file=f)
            print('True coefs for these features: ', true_coefs, file=f)
            print('Current coef error: ', coef_error, file=f)
            print('Init coef error: ', init_coef_error, file=f)
        coef_errors.append(coef_error)
        df = pd.DataFrame(coef_errors)
        df.to_csv(os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'Parameter_errors.csv'), header=False)
        plt.plot(torch.tensor(coef_errors),
                 'deepskyblue')  # , label='Error on coef')
        plt.xlim(xmin=0)
        # plt.ylim(0., 17)
        # plt.title('Error on coefficients of LS with current features')
        plt.xlabel('Number of cycles')
        plt.ylabel(r'RMSE')
        plt.savefig(
            os.path.join(
                '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                    loop),
                'Parameter_errors.pdf'), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        if loop == 0:
            state_estimation_RMSE = [
                RMS(xtraj - xtraj_estim[:, :xtraj.shape[1]]).numpy()]
        else:
            state_estimation_RMSE.append(
                RMS(xtraj - xtraj_estim[:, :xtraj.shape[1]]).numpy())
        df = pd.DataFrame(state_estimation_RMSE)
        df.to_csv(os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'State_estimation_RMSE.csv'), header=False)
        plt.plot(state_estimation_RMSE, 'deepskyblue')  # , label='RMSE')
        plt.xlim(xmin=0)
        plt.ylim(0.01, 0.12)
        # plt.title('State estimation RMSE')
        plt.xlabel('Number of cycles')
        plt.ylabel('RMSE')
        plt.savefig(
            os.path.join(
                '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                    loop),
                'State_estimation_RMSE.pdf'), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        if config.prior_mean is not None:
            plt.plot(reshape_dim1(reg.predict(sigma(X.numpy()))) +
                config.prior_mean(X, U, config.prior_kwargs).numpy(),
                     label='xi predicted by LS')
            plt.plot(torch.squeeze(Y + config.prior_mean(X, U,
                                                         config.prior_kwargs)),
                     label='xi given by observer')
        else:
            plt.plot(reg.predict(sigma(X.numpy())), label='xi predicted by LS')
            plt.plot(Y, label='xi given by observer')
        plt.legend()
        plt.title('Predicted xi depending on x')
        if config.verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'X0_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.plot(sigma(X)[:, 0], reg.predict(sigma(X.numpy())),
                 label='xi predicted by LS')
        plt.plot(sigma(X)[:, 0], Y, label='xi given by observer')
        plt.legend()
        plt.title('Predicted xi depending on sigma(x)')
        if config.verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'Sigma0_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.plot(sigma(X)[:, 1], reg.predict(sigma(X.numpy())),
                 label='xi predicted by LS')
        plt.plot(sigma(X)[:, 1], Y, label='xi given by observer')
        plt.legend()
        plt.title(r'Predicted xi depending on $\sigma(\dot{x})$')
        if config.verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + config.system + '/Least_squares_sigma/Loop_' + str(
                loop),
            'Sigma1_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()

        # Rollout with current model
        if loop == 0:
            x0 = torch.tensor([0.2, 0.5])
            u0 = reshape_pt1([0, 0])
            rollout_span = [config.t0, config.tf]
            rollout_eval = torch.linspace(config.t0, config.tf,
                                          config.nb_samples)
        rollout_true, rollout_utraj, _ = simulate_dynamics(
            t_eval=rollout_eval, t0=rollout_span[0], dt=config.dt,
            init_control=u0, init_state=x0,
            dynamics=config.dynamics, controller=config.controller,
            process_noise_var=config.process_noise_var,
            method=config.simu_solver, dyn_config=config,
            discrete=config.discrete, verbose=config.verbose)
        if 'LS_Michelangelo' in config.system:
            # Continuous-time model, numerical solver
            def Michelangelo_dyns(t, x):
                u = config.controller(t, config, 0, u0)
                # xi = current_model(x, u, config)
                xi = torch.as_tensor(reg.predict(sigma(x.numpy())))
                A = torch.tensor([[0., 1], [0., 0]])
                B = torch.tensor([[0], [1.]])
                ABmult = torch.matmul(A, reshape_pt1_tonormal(x)) + \
                         torch.matmul(B, reshape_pt1_tonormal(xi + u))
                return ABmult


            rollout_estim = odeint(Michelangelo_dyns, y0=x0, t=rollout_eval,
                                   **config.solver_options)
            plot_traj(predicted_mean_traj=rollout_estim, true_mean=rollout_true,
                      time=rollout_eval, subfolder=os.path.join(
                    '../Figures/' + config.system + f'/Least_squares_sigma/Loop_{loop}'))
        elif 'LS_justvelocity_highgain' in config.system:
            # Continuous model (discrete model is differentiated then num solver)
            def discrete_to_continuous_justvelocity_dyns(t, x):
                u = config.controller(t, config, 0, u0)
                # xnext = current_model(x, u, config)
                xnext = torch.as_tensor(reg.predict(sigma(x.numpy())))
                # xdot = (xnext - x[-1]) / dt
                xdot = xnext / dt  # contains prior mean!
                A = torch.tensor([[0., 1], [0., 0]])
                B = torch.tensor([[0.], [1]])
                ABmult = torch.matmul(A, reshape_pt1_tonormal(x)) + \
                         torch.matmul(B, reshape_pt1_tonormal(xdot + u))
                return ABmult


            rollout_estim = odeint(discrete_to_continuous_justvelocity_dyns,
                                    y0=x0, t=rollout_eval,
                                    **config.solver_options)
            plot_traj(predicted_mean_traj=rollout_estim,
                      true_mean=rollout_true,
                      time=rollout_eval, name='Continuous_',
                      subfolder=os.path.join(
                          '../Figures/' + config.system + f'/Least_squares_sigma/Loop_{loop}'))

            # Discrete model (continuous part Euler, discrete part stays)
            rollout_estim2 = x0 * torch.ones_like(rollout_estim)
            for ti in range(len(rollout_estim2) - 1):
                x = rollout_estim2[ti]
                u = config.controller(rollout_eval[ti], config, 0, u0)
                A = torch.tensor([[0., 1], [0, 0.]])
                B = torch.tensor([[0], [1.]])
                # xnext_n = current_model(x, u, config)
                xnext_n = torch.as_tensor(reg.predict(sigma(x.numpy())))
                # xnext = x + dt * (torch.matmul(A, reshape_pt1_tonormal(x)) + u)
                continuous_part = lambda x, u: \
                    torch.matmul(A, reshape_pt1_tonormal(x)) + \
                    torch.matmul(B, reshape_pt1_tonormal(u))
                xnext = rk4(x, continuous_part, dt, False, 0.01, u)
                xnext += reshape_pt1(
                    torch.matmul(B, reshape_pt1_tonormal(xnext_n)))
                rollout_estim2[ti + 1] = xnext
            plot_traj(predicted_mean_traj=rollout_estim2,
                      true_mean=rollout_true,
                      time=rollout_eval, name='Discrete_',
                      subfolder=os.path.join(
                          '../Figures/' + config.system + f'/Least_squares_sigma/Loop_{loop}'))
        if loop == 0:
            rollout_RMSE = \
                [torch.linalg.norm(rollout_true - rollout_estim).numpy()]
        else:
            rollout_RMSE += \
                [torch.linalg.norm(rollout_true - rollout_estim).numpy()]
        df = pd.DataFrame(rollout_RMSE)
        df.to_csv(os.path.join(
            '../Figures/' + config.system + f'/Least_squares_sigma/Loop_{loop}',
            'rollout_RMSE.csv'), header=False)
        plt.plot(rollout_RMSE, 'deepskyblue')  # , label='Error on coef')
        plt.xlim(xmin=0)
        # plt.title('Error on coefficients of LS with current features')
        plt.xlabel('Number of cycles')
        plt.ylabel(r'RMSE')
        plt.savefig(
            os.path.join('../Figures/' + config.system +
                         f'/Least_squares_sigma/Loop_{loop}',
                         'rollout_RMSE.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()

    stop_log()
