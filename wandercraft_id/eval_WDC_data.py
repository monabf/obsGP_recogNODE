import os
import logging
import control
import pandas as pd
import numpy as np
import dill as pkl
import torch
import matplotlib.pyplot as plt
from scipy import signal

from NN_for_ODEs.NODE_utils import make_init_state_obs
from simulation.dynamics import dynamics_traj
from simulation.observers import dynamics_traj_observer
from simulation.observer_functions import EKF_ODE
from model_evaluation.plotting_rollouts import NODE_rollout
from utils.utils import reshape_pt1, reshape_dim1, interpolate_func, RMS


def plot_NODE_full_rollouts(NODE, dt, vars=['u', 'theta', 'w'],
                            lin_model=None, verbose=False, save=False):
    with torch.no_grad():
        data_folder = '../Data/Wandercraft_id/'
        rollout_folder = os.path.join(NODE.results_folder,
                                      'Rollouts_' + str(NODE.step),
                                      'Full_rollouts_' + vars[0])
        os.makedirs(rollout_folder, exist_ok=True)
        keys = vars.copy()
        vars = dict.fromkeys(vars)
        for key in vars.keys():
            data_file = data_folder + key + '_slice.pkl'
            df = pd.read_pickle(data_file)
            vars[key] = [torch.squeeze(torch.as_tensor(df.iloc[i]))
                         for i in range(len(df))]
        u_slice = vars[keys[0]]
        theta_slice = vars[keys[1]]
        w_slice = vars[keys[2]]
        x0_list = []
        utraj_list = []
        xtraj_list = []
        xtraj_estim_list = []
        for m in range(len(u_slice)):
        # for m in range(10, 11):  # TODO
            time = torch.arange(0, len(theta_slice[m]) * dt, dt)
            xtraj_true = torch.cat((reshape_dim1(theta_slice[m]), reshape_dim1(
                w_slice[m])), dim=1)
            utraj = reshape_dim1(u_slice[m])
            t_u = torch.cat((reshape_dim1(time), reshape_dim1(utraj)), dim=1)
            controller = interpolate_func(
                x=t_u, t0=time[0], init_value=reshape_pt1(utraj[0]))
            NODE_time = torch.arange(0, len(theta_slice[m]) * dt, NODE.dt)
            NODE_utraj = controller(NODE_time, t0=NODE_time[0],
                                    init_value=utraj[0])
            init_state_obs = make_init_state_obs(
                xtraj_true, NODE_utraj, None, NODE_time, NODE.config)
            x0 = reshape_pt1(NODE.init_state_model(init_state_obs))
            xtraj_estim = NODE.NODE_model.forward_traj(
                x0, controller, NODE_time[0], NODE_time, reshape_pt1(utraj[0]))
            x0_list.append(x0)
            utraj_list.append(utraj)
            xtraj_list.append(xtraj_true)
            xtraj_estim_list.append(xtraj_estim)
            if lin_model is not None:
                xtraj_lin = torch.as_tensor(dynamics_traj(
                    x0=x0, u=controller, t0=0., dt=NODE.dt,
                    init_control=reshape_pt1(utraj[0]),
                    discrete=NODE.discrete, version=lin_model,
                    meas_noise_var=0., process_noise_var=0.,
                    method='dopri5', t_eval=time, kwargs=NODE.config))
                y_lin = reshape_dim1(NODE.observe_data(xtraj_lin))
            for l in range(xtraj_true.shape[1]):
                name = f'Full_rollout_prediction{m}_{l}.pdf'
                plt.plot(time, xtraj_true[:, l], label='True', c='g')
                plt.plot(NODE_time, NODE.observe_data(xtraj_estim)[:,
                                         l],
                         label='NODE', c='orange')
                if lin_model is not None:
                    plt.plot(time, y_lin[:, l], label='Linear', c='r')
                plt.legend()
                plt.title('Full data trajectory')
                if save:
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')
        return x0_list, utraj_list, xtraj_list, xtraj_estim_list


def plot_NODE_lin_rollouts(nb_rollouts, NODE, rollout_list,
                           rollouts_title=None, lin_model=None, type='default',
                           verbose=False, save=False):
    with torch.no_grad():
        RMSE_output_list = torch.zeros((len(rollout_list), 1))
        for i in range(nb_rollouts):
            if rollouts_title is None:
                rollout_folder = os.path.join(NODE.results_folder,
                                              'Rollouts_' + str(NODE.step),
                                              'Rollout_' + str(i))
            else:
                rollout_folder = os.path.join(
                    NODE.results_folder, rollouts_title + '_' + str(
                        NODE.step), 'Rollout_' + str(i))
            data = pd.read_csv(os.path.join(
                rollout_folder, 'Predicted_mean_traj.csv'), sep=',',
                header=None)
            # NODE predictions
            xtraj_NODE = torch.as_tensor(
                data.drop(data.columns[0], axis=1).values)
            y_NODE = NODE.observe_data(xtraj_NODE)
            data = pd.read_csv(os.path.join(
                rollout_folder, 'True_traj.csv'), sep=',', header=None)
            y_true = torch.as_tensor(data.drop(data.columns[0], axis=1).values)
            init_state = reshape_pt1(xtraj_NODE[0])
            control_traj = reshape_pt1(rollout_list[i][1])
            rollout_length = len(control_traj)
            time = torch.arange(0., rollout_length * NODE.dt, step=NODE.dt)
            # Compute predictions of linear model
            if lin_model is not None:
                t_u = torch.cat((reshape_dim1(time),
                                 reshape_dim1(control_traj)), dim=1)
                controller = interpolate_func(
                    x=t_u, t0=time[0], init_value=reshape_pt1(control_traj[0]))
                if type == 'default':
                    # Open-loop rollouts
                    xtraj_lin = torch.as_tensor(dynamics_traj(
                        x0=init_state, u=controller, t0=0., dt=NODE.dt,
                        init_control=reshape_pt1(control_traj[0]),
                        discrete=NODE.discrete, version=lin_model,
                        meas_noise_var=0., process_noise_var=0.,
                        method='dopri5', t_eval=time, kwargs=NODE.config))
                elif type == 'EKF':
                    # EKF rollouts
                    if NODE.config.get('prior_kwargs').get(
                            'EKF_added_meas_noise_var') is None:
                        meas_noise_var = NODE.true_meas_noise_var
                    else:
                        meas_noise_var = \
                            NODE.config.get('prior_kwargs').get(
                                'EKF_added_meas_noise_var')
                    if meas_noise_var != 0:
                        y_true += torch.normal(0, np.sqrt(
                            meas_noise_var), size=y_true.shape)
                    if NODE.config.get('prior_kwargs').get(
                            'EKF_observe_data') is not None:
                        logging.info('New measurement function for testing EKF')
                        y_EKF = NODE.config.get('prior_kwargs').get(
                            'EKF_observe_data')(y_true)
                    else:
                        y_EKF = y_true
                    t_y = torch.cat(
                        (reshape_dim1(time), reshape_dim1(y_EKF)), dim=1)
                    measurement = interpolate_func(
                        x=t_y, t0=time[0], init_value=reshape_pt1(y_EKF[0]))
                    covar0 = NODE.config.get('prior_kwargs').get(
                        'EKF_init_covar')
                    init_state = torch.cat(
                        (init_state, reshape_pt1(torch.flatten(covar0))),
                        dim=1)  # EKF state = (x, covar)
                    observer = EKF_ODE(init_state.device, NODE.config)
                    xtraj_lin = dynamics_traj_observer(
                        x0=init_state, u=controller, y=measurement, t0=time[0],
                        dt=NODE.dt, init_control=reshape_pt1(control_traj[0]),
                        discrete=False, version=observer, t_eval=time,
                        GP=lin_model, kwargs=NODE.config)
                    xtraj_lin = reshape_pt1(xtraj_lin[:, :NODE.n])  # no covar
                y_lin = reshape_dim1(NODE.observe_data(xtraj_lin))

                # RMSE of the linear model + save
                RMSE_output = RMS(NODE.scaler_Y.transform(y_lin) -
                                  NODE.scaler_Y.transform(y_true))
                RMSE_output_list[i] = RMSE_output
                if save:
                    filename = 'Linear_predicted_traj_mean.csv'
                    file = pd.DataFrame(reshape_pt1(xtraj_lin.cpu().numpy()))
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                    filename = 'Linear_RMSE.csv'
                    file = pd.DataFrame(reshape_pt1(RMSE_output))
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
            # Plot
            for k in range(xtraj_NODE.shape[1]):
                name = 'Lin_rollout_model_predictions' + str(k) + '.pdf'
                # plt.plot(time, xtraj_NODE[:, k], 'b',
                #          label='NODE', alpha=0.7)
                if lin_model is not None:
                    # plt.plot(time, xtraj_lin[:, k], c='r',
                    #          label='Linear model', alpha=0.9)
                    plt.plot(time, xtraj_lin[:, k], c='orange',
                             label='Linear model', alpha=0.9)
                plt.title('Predicted test trajectory')
                plt.legend()
                plt.xlabel('Time steps')
                plt.ylabel('State')
                if save:
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')
            for k in range(y_NODE.shape[1]):
                name = 'Lin_rollout_output_predictions' + str(k) + '.pdf'
                plt.plot(time, y_true[:, k], 'g', label='True')
                # plt.plot(time, y_NODE[:, k], label='NODE', c='orange',
                #          alpha=0.9)
                if lin_model is not None:
                    # plt.plot(time, y_lin[:, k], 'r', label='Linear', alpha=0.9)
                    plt.plot(time, y_lin[:, k], c='orange', label='Linear', alpha=0.9)
                plt.title('True and predicted output of test '
                          'trajectory')
                plt.legend()
                plt.xlabel('Time steps')
                plt.ylabel('Output')
                if save:
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')
    return torch.mean(reshape_pt1(RMSE_output_list)).cpu()

def plot_NODE_lin_Bode(NODE, dt, lin_model=None, vars=['u', 'theta', 'w'],
                       verbose=False, save=False, two_deformation=False):
    # name = 'Bode_diagram.pdf'
    # # Plot Bode of linear model
    # omega = np.logspace(-1.5, 3, 500)
    # control.bode_plot(lin_model.G[0, 0], plot=True, dB=True, Hz=True,
    #                   omega=omega, color="C0", linestyle="-.",
    #                   label='One deformation: u -> theta')
    # # control.bode_plot(lin_model.G[1, 0], plot=True, dB=True, Hz=True,
    # #                   omega=omega, color="C1", linestyle="-.",
    # #                   label='One deformation: u -> alpha')
    # control.bode_plot(
    #     control.TransferFunction.s * (
    #             lin_model.G[0, 0] + lin_model.G[1, 0]),
    #     plot=True, dB=True, Hz=True, omega=omega, color="C3",
    #     linestyle="-.", label='One deformation: u -> dq')
    # plt.suptitle('Open-loop transfer function for single deformation '
    #              'model.')

    # Plot Bode of NODE on training data
    x0_list, utraj_list, xtraj_list, xtraj_estim_list = \
        plot_NODE_full_rollouts(NODE, dt, vars=vars, verbose=verbose)
    fund_freqs = []
    for m in range(10, 11):  # TODO
        m = 0
    # for m in range(len(utraj_list)):
        u = utraj_list[m].view(-1,)
        N = len(u)
        freq = np.arange(0, N) / N / dt
        U = np.fft.fft(u)
        Uabs = np.abs(U)
        peaks, properties = signal.find_peaks(Uabs, height=1e5)
        fund_freq = freq[peaks[0]]
        print(m, fund_freq, len(u), len(peaks))
        fund_freqs.append(fund_freq)
        if len(peaks) > 2:
            print(f'More than 2 peaks detected for slice {m}, threshold wrong?')
        theta = xtraj_list[m][:, 0].view(-1,)
        peaks, properties = signal.find_peaks(theta, prominence=1e-3)
        time = np.arange(0, len(theta) * dt, dt)
        plt.plot(time, theta)
        plt.plot(time[peaks], theta[peaks], "x")
        print(m, fund_freq, properties['prominences'].min(),
              properties['prominences'].max())
        plt.show()
        plt.close('all')

    # # Get experimental Bode points from pickle
    # data = pkl.load(open(
    #     '../Data/Wandercraft_id/two_deformations/20200925T110815Z_LogFile_openloop.pickle',
    #     'rb'))
    # gain_encoder = [20 * np.log10(a / at) for a, at in
    #                 zip(data["a_encoder"], data["a_torque"])]
    # gain_gyro = [20 * np.log10(a / at) for a, at in
    #              zip(data["a_gyro"], data["a_torque"])]
    # data["phase_gyro"] = [p - 360 for p in data["phase_gyro"]]
    # plt.scatter(data["frequency"], data["phase_encoder"], color="C0")
    # plt.scatter(data["frequency"], data["phase_gyro"], color="C3")
    # # Plot points above 30 Hz in gray.
    # freq = [f for f in data["frequency"] if f > 31]
    # enc = [p for p, f in zip(data["phase_encoder"], data["frequency"]) if
    #        f > 31]
    # gyro = [p for p, f in zip(data["phase_gyro"], data["frequency"]) if
    #         f > 31]
    # plt.scatter(freq, enc, color="#000000")
    # plt.scatter(freq, gyro, color="#a0a0a0")
    # plt.sca(plt.gcf().axes[0])
    # plt.scatter(data["frequency"], gain_encoder, color="C0")
    # plt.scatter(data["frequency"], gain_gyro, color="C3")
    # enc = [p for p, f in zip(gain_encoder, data["frequency"]) if f > 31]
    # gyro = [p for p, f in zip(gain_gyro, data["frequency"]) if f > 31]
    # plt.scatter(freq, enc, color="#000000")
    # plt.scatter(freq, gyro, color="#a0a0a0")
    # plt.legend()
    # plt.gcf().axes[0].legend()
    # if save:
    #     plt.savefig(os.path.join(
    #         NODE.results_folder, 'Rollouts_' + str(NODE.step), name),
    #         bbox_inches='tight')
    # if verbose:
    #     plt.show()
    # plt.close('all')