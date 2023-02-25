import os
import sys

sys.path.append('.')

import torch
import numpy as np
import pandas as pd
import seaborn as sb

from simulation.simulation_functions import form_GP_data
from utils.utils import reshape_dim1, reshape_pt1

sb.set_style("whitegrid")


def approx_dynamics_from_data(x, control, X, U, Y):
    XU = torch.cat((X, U), dim=1)
    ynext = torch.zeros((len(x), Y.shape[1]))
    for i in range(len(x)):
        xu = torch.cat((reshape_pt1(x[i]), reshape_pt1(control[i])), dim=1)
        ynext[i] = reshape_pt1(Y[torch.argmin(torch.norm(XU - xu, dim=1))])
    return ynext


def read_points(folder):
    # Load points field over which output T is recorded
    points = np.fromfile(os.path.join(
        folder, 'temperature_points.bin'), dtype=np.float64)
    points = np.reshape(points[1:], (-1, 3))
    return points


def find_nearest_point(sensor, points):
    # Find point in points closest to sensor
    nearest_sensor_idx = np.argmin(
        np.linalg.norm(points - sensor.reshape((1, 3)), axis=1))
    nearest_sensor = points[nearest_sensor_idx]
    return nearest_sensor, nearest_sensor_idx


def read_whole_field(folder, nb_points, nb_scenarios=4):
    # Read whole field data T(x,t)
    whole_field = {}
    for i in range(1, nb_scenarios + 1):
        exc = pd.read_csv(os.path.join(folder, 'scenario' + str(
            i) + '/scenario' + str(i) + '_exc.csv'), sep=';')
        exc = exc.drop(exc.columns[0], axis=1).values
        whole_field['exc' + str(i)] = exc
        if i == 1:
            # in seconds, dt_before_subsampling = 1s
            time = np.arange(len(whole_field['exc1']))

        out = np.zeros((len(time), nb_points))
        for j in range(len(time)):
            out_current = np.fromfile(os.path.join(
                folder, 'scenario' + str(i) + '/scenario' + str(i) +
                        '_out/snapshot' + str(j + 1) + '.bin'),
                dtype=np.float64)
            out_current = np.reshape(out_current[1:], (1, -1))
            out[j, :] = out_current
        whole_field['out' + str(i)] = out
    return whole_field


def read_T_SVD_data(rom_folder, nb_scenarios=4):
    # Read T_SVD(t) of dim = nb modes, the evolution of the SVD modes over time
    T_SVD = {}
    for i in range(1, nb_scenarios + 1):
        df = pd.read_csv(os.path.join(rom_folder, 'scenario' + str(i) +
                                      '_mod.csv'), sep=';')
        df = df.drop(df.columns[0], axis=1).values
        T_SVD['out' + str(i)] = df
    return T_SVD


def read_T_SVD_afterROM_data(rom_folder, nb_scenarios=4):
    # Read T_SVD(t) of dim = nb modes, the evolution of the SVD modes over time
    T_SVD = {}
    for i in range(1, nb_scenarios + 1):
        df = pd.read_csv(os.path.join(rom_folder, 'scenario' + str(i) +
                                      '_mrm.csv'), sep=';')
        df = df.drop(df.columns[0], axis=1).values
        T_SVD['out' + str(i)] = df
    return T_SVD


def read_SVD_coefs(rom_folder, nb_modes=5):
    # Read SVD coefs of each point of the field in chosen SVD basis
    SVD_coefs = np.fromfile(os.path.join(rom_folder, 'SVDbasis.svd'),
                            dtype=np.float64)
    SVD_coefs = np.reshape(SVD_coefs[4:], (nb_modes, -1)).T
    return SVD_coefs


def compute_whole_field(T_SVD, SVD_coefs, nb_scenarios=4):
    # Reconstitute T(x,t) from T_SVD(t) for all scenarios
    whole_field_afterSVD = {}
    for i in range(1, nb_scenarios + 1):
        res = np.dot(T_SVD['out' + str(i)], SVD_coefs.T)
        whole_field_afterSVD['out' + str(i)] = res
    return whole_field_afterSVD


def form_Fluent_test_data(data, test_scenarios, config):
    # Form Fluent test data of type X = T_SVD(t), U = whole_field['exc'],
    # Y = T_SVD(t+1) for test with GPs, directly from ground truth,
    # remembering the indices of each scenario
    dimx = data['out' + str(test_scenarios[0])].shape[1]
    dimu = data['exc' + str(test_scenarios[0])].shape[1]
    X_test = np.zeros((0, dimx))
    U_test = np.zeros((0, dimu))
    Y_test = np.zeros((0, dimx))
    cut_idx = [0]

    if len(test_scenarios) == 0:
        cut_idx = []
        return X_test, U_test, Y_test, cut_idx

    for i in range(len(test_scenarios)):
        scenario = int(test_scenarios[i])
        xtraj = reshape_dim1(data['out' + str(scenario)])
        utraj = reshape_dim1(data['exc' + str(scenario)])
        test_system = config.system.replace('GP', 'Cross_val_test')
        x_test, u_test, y_test = form_GP_data(
            system=test_system, xtraj=xtraj, xtraj_estim=xtraj, utraj=utraj,
            meas_noise_var=config.true_meas_noise_var)
        X_test = np.concatenate((X_test, reshape_dim1(x_test)), axis=0)
        U_test = np.concatenate((U_test, reshape_dim1(u_test)), axis=0)
        Y_test = np.concatenate((Y_test, reshape_dim1(y_test)), axis=0)
        cut_idx += [len(X_test)]

    cut_idx = cut_idx[:-1]
    return X_test, U_test, Y_test, cut_idx
