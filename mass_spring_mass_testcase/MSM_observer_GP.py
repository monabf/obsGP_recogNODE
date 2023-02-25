import copy
import logging
import os
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import shutil
import sys

import dill as pkl
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sb
import torch
import GPy
from matplotlib import pyplot as plt

sys.path.append('.')

from simulation.controllers import sin_controller_1D
from simulation.dynamics import dynamics_traj
from simulation.dynamics_functions import MSM
from simulation.prior_means import Chain_integ_prior
from simulation.gain_adaptation_laws import simple_score_adapt_highgain, \
    Praly_highgain_adaptation_law
from simulation.observers import dim1_observe_data, \
    MSM_justvelocity_observer_adaptive_highgain_GP, \
    MSM_justvelocity_observer_highgain_GP, MSM_observer_Michelangelo_GP
from simulation.simulation_functions import simulate_dynamics, \
    simulate_estimations, \
    form_GP_data
from model_evaluation.plotting_functions import save_outside_data, \
    plot_outside_data
from utils.utils import reshape_pt1, reshape_dim1, interpolate_func, \
    reshape_dim1_tonormal, reshape_pt1_tonormal
from utils.utils import start_log, stop_log
from utils.config import Config
from GP_models.simple_GPyTorch_dyn import Simple_GPyTorch_Dyn

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

# Script to test quasi-linear system with observer, adding GP to learn
# nonlinear part, on a mass-spring-mass system put into the observable
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


def update_params_on_loop(system, config):
    if 'Duffing' in system:
        # omega = np.random.uniform(0, 2 * np.pi, 1)
        # dyn_config['omega'] = omega
        gamma = torch.rand(1) * (0.9 - 0.2) + 0.2
        config.gamma = gamma
    elif 'Pendulum' in system:
        omega = torch.rand(1) * (np.pi - 1) + 1
        config.omega = omega
        gamma = torch.rand(1) * (5 - 1) + 1
        config.gamma = gamma
    else:
        logging.warning('No parameter update defined for this system')
    return config


if __name__ == '__main__':
    start_log()

    # General params
    config = Config(system='Continuous/Mass_spring_mass/Discrete_model/'
                           'GP_Michelangelo_highgain_observer_noisy_inputs',
                    nb_samples=250,  # TODO
                    t0_span=0,
                    tf_span=15,
                    t0=0,
                    tf=15,
                    hyperparam_optim='fixed_hyperparameters',
                    true_meas_noise_var=1e-8,
                    process_noise_var=1e-8,
                    meas_noise_var=1,
                    simu_solver='dopri5',  # good solver needed for z
                    solver_options={'rtol': 1e-8, 'atol': 1e-8},
                    nb_loops=10,
                    nb_rollouts=10,  # TODO
                    rollout_length=200,
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
                    use_GPy=False,  # TODO
                    plot_output=False)
    t_eval = torch.linspace(config.t0, config.tf, config.nb_samples)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))

    # System params
    if 'Continuous/Mass_spring_mass' in config.system:
        config.update(dict(
            discrete=False,
            m1=1,
            m2=1,
            k1=0.3,
            k2=0.1,
            gamma=0.4,
            omega=1.2, dt=config.dt,
            dt_before_subsampling=0.001,
            dynamics_function=MSM,
            controller=sin_controller_1D,
            init_control=torch.tensor([[0.]])))
        dynamics = config.dynamics_function(config.cuda_device, config)
        config.update(dict(dynamics=dynamics,
                           dynamics_x=dynamics.mass_spring_mass_dynamics_x))
        # init_state_x = torch.tensor([[0., -0.01, 0.1, 0.01]])  # TODO
        # Random initial state
        xmin = torch.tensor([0., -0.005, 0.1, -0.005])
        xmax = torch.tensor([0.1, 0.005, 0.2, 0.005])
        init_state_x = reshape_pt1(torch.rand((1, 4)) * (xmax - xmin) + xmin)
        init_state = config.dynamics.mass_spring_mass_xtoz(
            init_state_x.clone(), config)
        init_state_estim = torch.cat((reshape_pt1(init_state[0, 0]),
                                      reshape_pt1(torch.tensor(
                                          [[0.] * (init_state.shape[1] - 1)]))),
                                     dim=1)
        config.update(dict(init_state_x=init_state_x,
                           init_state=init_state,
                           init_state_estim=init_state_estim))
        # Unstable limit cycle: needs g=50 or so with fhat=f to compensate
        #     init_state=reshape_pt1(torch.tensor([[
        #         0.06328351129370620, -0.4068801516907030, 0.4852631041981480,
        #         0.3863899972738590]])),
        #     init_state_estim=reshape_pt1(torch.tensor([[0., 0, 0, 0]])),
        #     init_control=reshape_pt1([0])))
        # config.update(dict(
        #     init_state_x=mass_spring_mass_ztox(config.init_state, config)))
        if 'GP_Michelangelo' in config.system:
            config.update(dict(
                observer_function=MSM_observer_Michelangelo_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10.,
                                                 'k1': 5.,
                                                 'k2': 5.,
                                                 'k3': 5.,
                                                 'k4': 2.,
                                                 'k5': 1}},
                              # 'observer_gains': {'g': 14.,  # TODO comparison
                              #                    'k1': 2.39433418,
                              #                    'k2': 2.67532355,
                              #                    'k3': 1.70816497,
                              #                    'k4': 0.61348767,
                              #                    'k5': 0.0979263}},
                saturation=np.array([-1, 1]),
                observer_prior_mean=None,
                prior_mean=None,
                prior_mean_deriv=None,
                init_state_estim=torch.cat((
                    config.init_state_estim, torch.tensor([[0.]])), dim=1)))
        elif 'GP_justvelocity_highgain' in config.system:
            config.update(dict(
                observer_function=MSM_justvelocity_observer_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10.,
                                                 'k1': 5.,
                                                 'k2': 5.,
                                                 'k3': 3.,
                                                 'k4': 1.},
                              # 'observer_gains': {'g': 14.,  # TODO comparison
                              #                    'k1': 1.96282935,
                              #                    'k2': 1.73371458,
                              #                    'k3': 0.79403003,
                              #                    'k4': 0.15585455},
                              'backward_observer_gains': {'g': 10.,
                                                          'k1': 5.,
                                                          'k2': -5.,
                                                          'k3': 3.,
                                                          'k4': -1.}}))
            if config.continuous_model:
                config.update(dict(saturation=np.array([-1, 1]),
                                   observer_prior_mean=None,
                                   prior_mean=None,
                                   prior_mean_deriv=None))
            else:
                prior = Chain_integ_prior(config.cuda_device,
                                          config.prior_kwargs)
                config.update(dict(
                    saturation=np.array([-1, 1]),
                    observer_prior_mean=None,
                    prior_mean=prior.discrete_justvelocity,
                    prior_mean_deriv=None))
        elif 'GP_justvelocity_adaptive_highgain' in config.system:
            config.update(dict(
                observer_function=MSM_justvelocity_observer_adaptive_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains':
                                  {'g': 12.,
                                   'g0': 12.,  # TODO
                                   'k1': 5,
                                   'k2': 5,
                                   'k3': 3,
                                   'k4': 1,
                                   'p1': 4,
                                   'p2': 0.002,
                                   'b': 0.2,
                                   'n': config.init_state.shape[1],
                                   'adaptation_law':
                                       Praly_highgain_adaptation_law}},
                saturation=np.array([-1, 1])))
            prior = Chain_integ_prior(config.cuda_device, config.prior_kwargs)
            config.update(dict(
                observer_prior_mean=None,
                prior_mean=prior.discrete_justvelocity,
                prior_mean_deriv=None,
                init_state_estim=torch.cat((
                    config.init_state_estim,
                    torch.tensor([[config.prior_kwargs[
                                       'observer_gains'].get('g0')]])), dim=1)))
        elif 'No_observer' in config.system:
            config.update(dict(observer_function=None,
                               observer_prior_mean=None,
                               prior_kwargs={'dt': config.dt,
                                             'dt_before_subsampling': 0.001,
                                             'observer_gains': {'g': 10,
                                                                'k1': 5,
                                                                'k2': 5,
                                                                'k3': 3,
                                                                'k4': 1}}))
        # Create kernel
        if config.get('gamma') == 0:
            input_dim = config.init_state.shape[1]
            no_control = True
        else:
            input_dim = config.init_state.shape[1] + \
                        config.init_control.shape[1]
            no_control = False
        # For each part of the likelihood (Gaussian noise, scale kernel,
        # RBF kernel), define a prior distribution which will influence the
        # hyperparameter optimization, and an initial value
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # noise_prior=gpytorch.priors.NormalPrior(config.meas_noise_var, 10.))
        likelihood.noise_covar.noise = torch.tensor(config.meas_noise_var)
        likelihood.noise = torch.tensor(config.meas_noise_var)
        lengthscale_prior = gpytorch.priors.MultivariateNormalPrior(
            torch.tensor([150., 150., 1.5, 2.5, 2.5]),
            torch.diag(torch.tensor([150., 150., 15, 25, 25])))
        outputscale_prior = gpytorch.priors.NormalPrior(3.5, 35)
        hypers = {'base_kernel.lengthscale': lengthscale_prior.mean,
                  'outputscale': outputscale_prior.mean}
        if config.use_GPy:
            kernel = GPy.kern.RBF(input_dim=input_dim, variance=3.5,
                                  lengthscale=np.array(
                                      [150, 150, 1.5, 2.5, 2.5]),
                                  ARD=True)  # TODO test Steve
            kernel.unconstrain()
            kernel.variance.set_prior(GPy.priors.Gaussian(3.5, 3.5))
            kernel.lengthscale.set_prior(
                GPy.priors.MultivariateGaussian(
                    np.array([150, 150, 1.5, 2.5, 2.5]),
                    np.diag([150, 150, 1.5, 2.5, 2.5])))
        else:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim,
                lengthscale_prior=lengthscale_prior),
                outputscale_prior=outputscale_prior)
            kernel.initialize(**hypers)
        config.update(dict(observe_data=dim1_observe_data,
                           observer=config.observer_function(
                               config.cuda_device, config),
                           constrain_u=[-config.get('gamma'),
                                        config.get('gamma')],
                           constrain_x=[],
                           grid_inf=-0.5,
                           grid_sup=0.5,
                           input_dim=input_dim,
                           no_control=no_control,
                           kernel=kernel,
                           likelihood=likelihood))
    else:
        raise Exception('Unknown system')

    # Set derivative_function for continuous models
    if config.continuous_model:
        if config.prior_mean:
            logging.warning(
                'A prior mean has been defined for the GP though '
                'a continuous model is being used. Check this is '
                'really what you want to do, as a prior mean is '
                'often known for discrete models without being '
                'available for continuous ones.')


        def derivative_function(X, U, y_observed, GP):
            X = reshape_pt1(X)
            u = lambda t, kwargs, t0, init_control: reshape_pt1(U)[t]
            y = lambda t, kwargs: reshape_pt1(y_observed)[t]
            Xdot = torch.tensor([config.observer(t, X[t], u, y, config.t0,
                                                 config.init_control, GP,
                                                 config) for t in
                                 range(len(X))])
            Xdot = Xdot.reshape(X.shape)
            return Xdot.reshape(X.shape)


        config.update(dict(derivative_function=derivative_function))
    else:
        config.update(dict(derivative_function=None))

    # Check valid config
    if (config.m1 < 0) or (config.m2 < 0) or (config.k1 < 0) or (config.k2 < 0):
        raise ValueError('All physical parameters should be positive for this '
                         'solution of the MSM system to be valid: '
                         'http://eqworld.ipmnet.ru/en/solutions/ae/ae0103.pdf')

    u0 = config.init_control.clone()
    t0_init = copy.deepcopy(config.t0)


    # Define true dynamics
    def true_dynamics(x, control):
        device = x.device
        t_u = torch.cat((reshape_dim1(torch.arange(
            len(control), device=device)), control), dim=1)
        # Put controller data to cuda for interpolation
        if torch.cuda.is_available():
            t_u = t_u.cuda()

        true_controller = interpolate_func(x=t_u, t0=config.t0,
                                           init_value=config.init_control)

        if ('justvelocity' in config.system) and (
                'Mass_spring_mass' in config.system):
            if config.continuous_model:
                xnext = config.dynamics_function(
                    t=t0_init, z=x, u=true_controller, t0=t0_init,
                    init_control=control,
                    process_noise_var=config.process_noise_var,
                    kwargs=config)[:, -1]
            else:
                xnext = dynamics_traj(
                    x0=reshape_pt1(x), u=true_controller, t0=t0_init,
                    dt=config.dt, init_control=u0,
                    version=config.dynamics, meas_noise_var=0,
                    process_noise_var=config.process_noise_var,
                    method=config.simu_solver, t_eval=[config.dt],
                    kwargs=config)[:, -1]
        elif ('Michelangelo' in config.system) and (
                'Mass_spring_mass' in config.system):
            m1 = config.get('m1')
            m2 = config.get('m2')
            k1 = config.get('k1')
            k2 = config.get('k2')
            z = reshape_pt1(x)
            z3 = reshape_pt1(z[:, 2])
            u = control
            v = reshape_pt1_tonormal(
                config.dynamics.mass_spring_mass_v(z, config))
            vdot = reshape_pt1_tonormal(
                config.dynamics.mass_spring_mass_vdot(z, config))
            xnext = reshape_pt1(
                k1 * (m1 * m2) * (u - (m1 + m2) * z3) + (3 * k2)
                / (m1 * m2) * (u - (m1 + m2) * z3) * v ** 2
                + (6 * k2) / m1 * v * vdot ** 2)
        elif (('Michelangelo' in config.system) or (
                'justvelocity_highgain' in config.system)) \
                and not any(
            k in config.system for k in ('Duffing', 'Harmonic_oscillator',
                                         'Pendulum', 'VanderPol')):
            raise Exception('No ground truth has been defined.')
        else:
            xnext = dynamics_traj(
                x0=reshape_pt1(x), u=true_controller, t0=t0_init,
                dt=config.dt, init_control=u0, version=config.dynamics,
                meas_noise_var=0, process_noise_var=config.process_noise_var,
                method=config.simu_solver, t_eval=[config.dt],
                kwargs=config.dyn_kwargs)
        return xnext


    config.update(dict(true_dynamics=true_dynamics))

    # Loop to create data and learn GP several times
    for loop in range(config.nb_loops):

        # Adapt parameters on loop start if necessary
        if loop > 0:
            # Update params and initial states after the first pass
            if config.restart_on_loop:
                config = update_params_on_loop(config.system, config)
                t0 = config.t0
                tf = config.tf
                t0_span = config.t0_span
                tf_span = config.tf_span
            else:
                init_state_x = reshape_pt1(xtraj_orig_coord[-1].clone())
                init_state = reshape_pt1(xtraj[-1].clone())
                init_control = reshape_pt1(utraj[-1].clone())
                init_state_estim = reshape_pt1(xtraj_estim[-1].clone())
                tf_before = config.tf
                tf_span = tf_before + (config.tf_span - config.t0_span)
                t0_span = tf_before
                tf = tf_before + (config.tf - config.t0)
                t0 = tf_before
                config.update(dict(t0=t0, tf=tf, t0_span=t0_span,
                                   tf_span=tf_span, init_state_x=init_state_x,
                                   init_state=init_state,
                                   init_control=init_control,
                                   init_state_estim=init_state_estim))
            t_eval = torch.linspace(t0, tf, config.nb_samples)
            # Put all tensor data on cuda
            if torch.cuda.is_available():
                for key, val in config.items():
                    if torch.is_tensor(val):
                        print(key)
                        val = val.cuda()

            # Update observer gain
            if config.batch_adaptive_gain:
                if 'simple_score_posdist_lastbatch' in \
                        config.batch_adaptive_gain:
                    gain = config.prior_kwargs['observer_gains'].get('g')
                    score = torch.norm(
                        reshape_dim1_tonormal(xtraj_estim[:, 0] - xtraj[:, 0]))
                    previous_idx = int(np.min([loop, 2]))
                    (base_path, previous_loop) = os.path.split(
                        dyn_GP.results_folder)
                    previous_results_folder = os.path.join(
                        base_path, 'Loop_' + str(loop - previous_idx))
                    previous_xtraj_estim = pd.read_csv(os.path.join(
                        previous_results_folder,
                        'Data_outside_GP/xtraj_estim.csv'),
                        sep=',', header=None)
                    previous_xtraj_estim = previous_xtraj_estim.drop(
                        previous_xtraj_estim.columns[0], axis=1).values
                    previous_xtraj = pd.read_csv(os.path.join(
                        previous_results_folder, 'Data_outside_GP/xtraj.csv'),
                        sep=',', header=None)
                    previous_xtraj = previous_xtraj.drop(
                        previous_xtraj.columns[0], axis=1).values
                    previous_score = torch.norm(reshape_dim1_tonormal(
                        previous_xtraj_estim[:, 0] - previous_xtraj[:, 0]))
                    new_gain = simple_score_adapt_highgain(gain, score,
                                                           previous_score)
                    config.prior_kwargs['observer_gains']['g'] = new_gain
                    gain_time = torch.cat(
                        (gain_time, torch.tensor([new_gain])))
                elif 'simple_score_' in config.batch_adaptive_gain:
                    param = \
                        config.batch_adaptive_gain.split('simple_score_', 1)[1]
                    gain = config.prior_kwargs['observer_gains'].get('g')
                    score = dyn_GP.variables[param][-1, 1]
                    previous_idx = int(np.min([loop, 2]))
                    previous_score = dyn_GP.variables[param][-previous_idx, 1]
                    new_gain = simple_score_adapt_highgain(gain, score,
                                                           previous_score)
                    config.prior_kwargs['observer_gains']['g'] = new_gain
                    gain_time = torch.cat(
                        (gain_time, torch.tensor([new_gain])))
                elif config.batch_adaptive_gain == 'change_last_batch':
                    if loop == config.nb_loops - 1:
                        new_gain = 3
                        config.prior_kwargs['observer_gains']['g'] = new_gain
                        gain_time = torch.cat(
                            (gain_time, torch.tensor([new_gain])))
                else:
                    logging.error(
                        'This adaptation law for the observer gains has '
                        'not been defined.')

        # Simulate system in x
        xtraj_orig_coord, utraj, t_utraj = \
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
        if config.verbose:
            # Plot trajectories
            plt.plot(xtraj_orig_coord[:, 0], label='x1')
            plt.plot(xtraj_orig_coord[:, 2], label='x2')
            plt.title('Positions over time')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            plt.plot(xtraj_orig_coord[:, 1], label=r'$\dot{x_1}$')
            plt.plot(xtraj_orig_coord[:, 3], label=r'$\dot{x_2}$')
            plt.title('Velocities over time')
            plt.xlabel('t')
            plt.ylabel(r'$\dot{x}$')
            plt.legend()
            plt.show()
            plt.plot(xtraj_orig_coord[:, 0], xtraj_orig_coord[:, 1], label='x1')
            plt.plot(xtraj_orig_coord[:, 2], xtraj_orig_coord[:, 3], label='x2')
            plt.title('Phase portraits')
            plt.xlabel('x')
            plt.ylabel(r'$\dot{x}$')
            plt.legend()
            plt.show()
            plt.plot(utraj, label='u')
            plt.title('Control trajectory')
            plt.xlabel('t')
            plt.ylabel('u')
            plt.legend()
            plt.show()

        # Simulate corresponding system in z
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
            plt.plot(xtraj_orig_coord[:, 0], label='x')
            plt.plot(config.dynamics.mass_spring_mass_ztox(xtraj, config)[:, 0],
                     label='T*(''z)')
            plt.plot(config.dynamics.mass_spring_mass_ztox(
                config.dynamics.mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 0], label='T*(T(x))')
            plt.legend()
            plt.show()
            plt.plot(xtraj_orig_coord[:, 1], label='x')
            plt.plot(config.dynamics.mass_spring_mass_ztox(xtraj, config)[:, 1],
                     label='T*(z)')
            plt.plot(config.dynamics.mass_spring_mass_ztox(
                config.dynamics.mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 1], label='T*(T(x))')
            plt.legend()
            plt.show()
            plt.plot(xtraj_orig_coord[:, 2], label='x')
            plt.plot(config.dynamics.mass_spring_mass_ztox(xtraj, config)[:, 2],
                     label='T*(z)')
            plt.plot(config.dynamics.mass_spring_mass_ztox(
                config.dynamics.mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 2], label='T*(T(x))')
            plt.legend()
            plt.show()
            plt.plot(xtraj_orig_coord[:, 3], label='x')
            plt.plot(config.dynamics.mass_spring_mass_ztox(xtraj, config)[:, 3],
                     label='T*(z)')
            plt.plot(config.dynamics.mass_spring_mass_ztox(
                config.dynamics.mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 3], label='T*(T(x))')
            plt.legend()
            plt.show()

        # Observe data: only position, observer reconstitutes velocity
        # Get observations over t_eval and simulate xhat only over t_eval
        if loop == 0:
            observer_prior_mean = config.observer_prior_mean
        else:
            observer_prior_mean = dyn_GP
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

        # Initialize or re-initialize GP
        if loop == 0:
            dyn_GP = Simple_GPyTorch_Dyn(X, U, Y, config)
        else:
            (base_path, previous_loop) = os.path.split(dyn_GP.results_folder)
            new_results_folder = os.path.join(base_path, 'Loop_' + str(loop))
            os.makedirs(new_results_folder, exist_ok=False)
            dyn_GP.set_results_folder(new_results_folder)
            dyn_GP.set_config(config)

        # Set up outside data to save
        data_to_save = {
            'xtraj_orig_coord': xtraj_orig_coord, 'xtraj': xtraj,
            'xtraj_estim': xtraj_estim, 'transfo_ztox_xtoz_xtraj_orig_coord':
                config.dynamics.mass_spring_mass_ztox(
                    config.dynamics.mass_spring_mass_xtoz(
                        xtraj_orig_coord, config), config),
            'transfo_ztox_xtraj':
                config.dynamics.mass_spring_mass_ztox(xtraj, config),
            'y_observed': y_observed}
        if config.batch_adaptive_gain:
            gain_time = torch.tensor(config.prior_kwargs['observer_gains']['g'])
            data_to_save.update({'gain_time': gain_time})
        elif 'adaptive' in config.system:
            output_error = reshape_dim1(
                torch.square(xtraj[:, 0] - xtraj_estim[:, 0]))
            if loop == 0:
                gain_time = reshape_dim1(xtraj_estim[:, -1])
            else:
                gain_time = torch.cat((
                    gain_time, reshape_dim1(xtraj_estim[:, -1])))
            data_to_save.update(
                {'gain_time': gain_time, 'output_error': output_error})
            dyn_GP.config.prior_kwargs['observer_gains']['g'] = gain_time[-1]

        # Train GP with estimated trajectory, evaluate and save results
        if config.monitor_experiment:
            save_outside_data(dyn_GP, data_to_save)
            plot_outside_data(dyn_GP, data_to_save)
        if loop == 0:
            dyn_GP.learn()

            # Run rollouts using only priors, before learning (step=-1)
            rollouts_folder = os.path.join(dyn_GP.results_folder, 'Rollouts_0')
            new_rollouts_folder = os.path.join(dyn_GP.results_folder,
                                               'Rollouts_-1')
            shutil.copytree(rollouts_folder, new_rollouts_folder)
            old_step, dyn_GP.step = dyn_GP.step, 0
            old_sample_idx, dyn_GP.sample_idx = dyn_GP.sample_idx, 0
            if 'adaptive' in config.system:
                # Do not adapt observer gains for closed-loop rollouts
                dyn_GP.config.prior_kwargs['observer_gains']['g'] = \
                    config.prior_kwargs['observer_gains']['g0']
                dyn_GP.evaluate_closedloop_rollouts(
                    config.observer.call_noadapt, config.observe_data,
                    no_GP_in_observer=True)
                dyn_GP.config.prior_kwargs['observer_gains']['g'] = \
                    gain_time[-1]
            else:
                dyn_GP.evaluate_closedloop_rollouts(
                    config.observer, config.observe_data,
                    no_GP_in_observer=True)
            if config.prior_mean:
                # Also run open-loop rollouts with prior before learning
                dyn_GP.evaluate_rollouts(only_prior=True)
            dyn_GP.step = old_step
            dyn_GP.sample_idx = old_sample_idx

        else:
            dyn_GP.learn(new_X=X, new_Y=Y, new_U=U)
        dyn_GP.save()
        if 'adaptive' in config.system:
            # Do not adapt observer gains for closed-loop rollouts
            dyn_GP.evaluate_closedloop_rollouts(
                config.observer.call_noadapt, config.observe_data)
        else:
            dyn_GP.evaluate_closedloop_rollouts(config.observer,
                                                config.observe_data)

    # At end of training, save GP model with pickle
    with open(dyn_GP.results_folder + '/GP_submodel.pkl', 'wb') as f:
        pkl.dump(dyn_GP.model, f, protocol=4)
    with open(dyn_GP.results_folder + '/GP_model.pkl', 'wb') as f:
        pkl.dump(dyn_GP, f, protocol=4)

    stop_log()
