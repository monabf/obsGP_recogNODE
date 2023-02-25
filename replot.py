# Copyright 2019 Max Planck Society. All rights reserved.
import logging
import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

from simulation.observers import dim1_observe_data, dim14_observe_data
from utils.utils import reshape_pt1, reshape_dim1, RMSE


# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rcdefaults()
# For manuscript
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
# # Previous papers
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 22})
# sb.set_style('whitegrid')



# For nice box plots https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots


def box_plots_vars_time(folder, variables=['RMSE'], avoid0=False,
                        errorbars=False):
    # List all folders = different experiments in current folder
    subfolders_unfiltered = [os.path.join(folder, o) for o in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, o))]
    subfolders = []
    for subfolder in subfolders_unfiltered:
        if ('Box_plots' in subfolder) or ('Ignore' in subfolder):
            continue  # Skip all subfolders containing Box_plots so avoid!
        subfolders += [subfolder]

    # Gather all variables to plot in dictionary
    vars = dict.fromkeys(variables)
    for key, val in vars.items():
        vars[key] = []
        for subfolder in subfolders:
            try:
                name = key + '.csv'
                data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                                   header=None)
                values = data.drop(data.columns[0], axis=1).values
                if values.shape[1] == 1:
                    values = np.concatenate(
                        (np.arange(len(values)).reshape(-1, 1),
                         values), axis=1)
                if values[-1, 1] > 50:
                    # Ignore folders with values to large (must be problem)
                    logging.warning('Experiment ' + str(subfolder) +
                                    ' was ignored because of unexpectedly '
                                    'large error values')
                    subfolders.remove(subfolder)
                    continue
                if avoid0:
                    # Avoid first value at sample_idx = 0
                    if 'Data_outside' not in key:
                        values = values[np.logical_not(values[:, 0] == 0)]
                vars[key].append(values)
            except FileNotFoundError:
                print(
                    'Files ' + os.path.join(subfolder, str(key)) + ' not found')

    # Make results folder
    results_folder = os.path.join(folder, 'Box_plots')
    os.makedirs(results_folder, exist_ok=True)  # Will overwrite!
    specs_path = os.path.join(results_folder, 'Specifications.txt')
    shutil.copy(os.path.join(subfolders[0], 'Loop_0/Specifications.txt'),
                results_folder)
    with open(specs_path, 'a') as f:
        print('\n', file=f)
        print('\n', file=f)
        print('\n', file=f)
        print('Box plots from ' + str(len(subfolders)) + ' experiments',
              file=f)
        print('Experiments not used: ' +
              str(list(set(subfolders_unfiltered) - set(subfolders))), file=f)

    # Compute mean and std for each variable, then plot
    vars_mean = dict.fromkeys(variables)
    vars_std = dict.fromkeys(variables)
    for key, val in vars.items():
        vars_mean[key] = np.mean(np.array(vars[key]), axis=0)
        vars_std[key] = np.std(np.array(vars[key]), axis=0)
        vars_quant1 = np.quantile(np.array(vars[key]), q=0.25)
        vars_quant2 = np.quantile(np.array(vars[key]), q=0.75)
        name = key.replace('/', '_') + '.pdf'  # avoid new dirs
        # plt.plot(vars_mean[key][:, 0], vars_mean[key][:, 1], 'deepskyblue')
        plt.plot(np.arange(len(vars_mean[key])), vars_mean[key][:, 1], 'deepskyblue')
        # if errorbars:
        #     errorevery = 1
        #     markevery = 1
        #     plt.errorbar(vars_mean[key][:, 0], vars_mean[key][:, 1],
        #                  yerr=vars_std[key][:, 1], fmt='o', ls='-',
        #                  c='deepskyblue', capsize=2, alpha=0.8,
        #                  errorevery=errorevery, markevery=markevery)
        # else:
        #     print(key, len(vars_mean[key]))
        #     print(vars_mean[key][:, 1])
        #     plt.fill_between(np.arange(len(vars_mean[key])),
        #         # vars_mean[key][:, 0],
        #                      vars_mean[key][:, 1] - 2 * vars_std[key][:, 1],
        #                      vars_mean[key][:, 1] + 2 * vars_std[key][:, 1],
        #                      facecolor='deepskyblue', alpha=0.2)
        #     # plt.fill_between(np.arange(len(vars_mean[key])),
        #     #                  vars_quant1,
        #     #                  vars_quant2,
        #     #                  facecolor='deepskyblue', alpha=0.2)
        # plt.xlim(xmin=vars_mean[key][0, 0])
        # if 'Closedloop_rollout_RMSE' in key:
        #     plt.ylim(bottom=0.63, top=0.7)
        # if 'Rollout_RMSE' in key:
        #     plt.ylim(bottom=0)
        # plt.title('Estimation error over time')
        plt.xlabel('Number of cycles')
        plt.ylabel('RMSE')
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.clf()
        plt.close('all')


def replot_vars_time(subfolder, variables=['RMSE'], avoid0=False):
    # From single subfolder and variables of type [time, values], replot
    # value by time
    variables_plot = dict.fromkeys(variables)
    for key in variables_plot:
        try:
            name = key + '.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            values = data.drop(data.columns[0], axis=1).values
            if values.shape[1] == 1:
                values = np.concatenate(
                    (np.arange(len(values)).reshape(-1, 1),
                     values), axis=1)
            if avoid0:
                # Avoid first value at sample_idx = 0
                if 'Data_outside' not in key:
                    values = values[np.logical_not(values[:, 0] == 0)]
            name = key + '.pdf'
            # plt.plot(values[:, 0], values[:, 1], 'deepskyblue')
            # plt.xlim(xmin=values[0, 0])
            if 'Parameter_errors' in key:
                plt.plot(np.arange(len(values)), values[:, 1], 'deepskyblue')
                plt.ylabel('Relative error')
                plt.ylim(-0.05, 1.105)
            elif 'State_estimation_RMSE' in key:
                plt.plot(np.arange(len(values)), values[:, 1], 'deepskyblue')
                plt.ylabel('RMSE')
                plt.ylim(0., 0.12)
            else:
                plt.plot(np.arange(len(values)), values[:, 1], 'deepskyblue')
                plt.ylabel('RMSE')
                plt.ylim(-50, 3500)
            # plt.plot(np.arange(len(values)), values[:, 1], 'deepskyblue')
            # plt.plot(np.arange(len(values)), values[:, 0], 'b')
            plt.xlim(xmin=0)
            # plt.title('Estimation error over time')
            plt.xlabel('Number of cycles')
            # plt.ylabel('RMSE')
            # plt.xlabel('Time steps')
            # plt.ylabel('Gain')
            plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
            plt.clf()
            plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + ' not found')


def replot_rollouts(subfolder, plots=[], plot_output=False,
                    observe_data=dim1_observe_data,
                    true_meas_noise_var=0., GP=False):
    # From rollout folder name, replot rollout
    variables_plot = dict.fromkeys(plots)
    for key in variables_plot:
        # True traj
        name = key + '/True_traj.csv'
        data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                           header=None)
        true_mean = data.drop(data.columns[0], axis=1).values
        time = np.arange(0, len(true_mean)) * 0.001
        try:
            # Open-loop trajs
            # name = key + '/Predicted_mean_traj.csv'
            name = key + 'Linear_predicted_traj_mean.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_mean_traj = data.drop(data.columns[0], axis=1).values
        #     if GP:
        #         name = key + '/Predicted_uppconf_traj.csv'
        #         data = pd.read_csv(os.path.join(subfolder, name), sep=',',
        #                            header=None)
        #         predicted_traj_uppconf = data.drop(data.columns[0], axis=1).values
        #         name = key + '/Predicted_lowconf_traj.csv'
        #         data = pd.read_csv(os.path.join(subfolder, name), sep=',',
        #                            header=None)
        #         predicted_traj_lowconf = data.drop(data.columns[0], axis=1).values
        #     for i in range(predicted_mean_traj.shape[1]):
        #         name = key + 'Rollout_model_predictions' + str(
        #             i) + '.pdf'
        #         plt.plot(time, true_mean[:, i], 'g', label='True')
        #         plt.plot(time, predicted_mean_traj[:, i],
        #                  label='Predicted', c='b', alpha=0.7)
        #         if GP:
        #             plt.fill_between(time,
        #                              predicted_traj_lowconf[:, i],
        #                              predicted_traj_uppconf[:, i],
        #                              facecolor='blue', alpha=0.2)
        #         plt.legend()
        #         # plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5),
        #         #            frameon=True)
        #         plt.xlim(xmin=time[0])
        #         plt.xlabel(r'$t$')
        #         plt.ylabel(r'$x_{}$'.format(i + 1))
        #         plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        #         plt.close('all')
        #     for i in range(predicted_mean_traj.shape[1] - 1):
        #         name = key + 'Rollout_phase_portrait' + str(i) + '.pdf'
        #         plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
        #                  label='True')
        #         plt.plot(predicted_mean_traj[:, i],
        #                  predicted_mean_traj[:, i + 1],
        #                  label='Predicted', c='b', alpha=0.7)
        #         if GP:
        #             plt.fill_between(predicted_mean_traj[:, i],
        #                              predicted_traj_lowconf[:, i + 1],
        #                              predicted_traj_uppconf[:, i + 1],
        #                              facecolor='blue', alpha=0.2)
        #         plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
        #                     marker='x', s=100, label='Initial state')
        #         plt.legend()
        #         # plt.legend(loc='lower right')
        #         plt.xlabel(r'$x_{}$'.format(i + 1))
        #         plt.ylabel(r'$x_{}$'.format(i + 2))
        #         plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        #         plt.close('all')

            if plot_output:
                # Traj of system output
                # y_observed = reshape_dim1(
                #     observe_data(torch.as_tensor(true_mean)))
                y_observed = true_mean  # if ground_truth_approx!
                if true_meas_noise_var != 0:
                    y_observed = reshape_pt1(y_observed + np.random.normal(
                        0, np.sqrt(true_meas_noise_var), y_observed.shape))
                for i in range(y_observed.shape[1]):
                    name = key + 'Rollout_output_predictions' + str(i) + '.pdf'
                    plt.plot(time[:10000], y_observed[:10000, i], 'g',
                             label='True')
                    plt.plot(time[:10000], observe_data(
                        torch.as_tensor(predicted_mean_traj))[:10000, i],
                             # label='Predicted', c='orange', alpha=0.9)
                             label='Predicted', c='b', alpha=0.7)
                    if i == 0:
                        plt.legend()
                    plt.xlim(xmin=time[0])
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'$y$')
                    plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                    plt.close('all')

                error = RMSE(torch.as_tensor(y_observed), observe_data(
                    torch.as_tensor(predicted_mean_traj)))
                with open(os.path.join(
                        subfolder, key, 'Output_RMSE.txt'), 'w') as f:
                    print(f'RMSE on predicted output = {error}', file=f)
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + ' openloop '
                                                                 'not found')

        # try:
        #     # Kalman trajs
        #     name = key + '/Predicted_mean_traj_kalman.csv'
        #     data = pd.read_csv(os.path.join(subfolder, name), sep=',',
        #                        header=None)
        #     predicted_mean_traj_kalman = data.drop(data.columns[0],
        #                                            axis=1).values
        #     if GP:
        #         name = key + '/Predicted_uppconf_traj_kalman.csv'
        #         data = pd.read_csv(os.path.join(subfolder, name), sep=',',
        #                            header=None)
        #         predicted_traj_uppconf_kalman = data.drop(data.columns[0],
        #                                                   axis=1).values
        #         name = key + '/Predicted_lowconf_traj_kalman.csv'
        #         data = pd.read_csv(os.path.join(subfolder, name), sep=',',
        #                            header=None)
        #         predicted_traj_lowconf_kalman = data.drop(data.columns[0],
        #                                                   axis=1).values
        #     for i in range(predicted_mean_traj_kalman.shape[1]):
        #         name = key + 'Kalman_rollout_model_predictions' + str(
        #             i) + '.pdf'
        #         plt.plot(time, true_mean[:, i], 'g', label='True')
        #         plt.plot(time, predicted_mean_traj_kalman[:, i],
        #                  label='Predicted', c='b', alpha=0.7)
        #         plt.fill_between(time,
        #                          predicted_traj_lowconf_kalman[:, i],
        #                          predicted_traj_uppconf_kalman[:, i],
        #                          facecolor='blue', alpha=0.2)
        #         plt.legend()
        #         plt.xlim(xmin=time[0])
        #         plt.xlabel(r'$t$')
        #         plt.ylabel(r'$x_{}$'.format(i + 1))
        #         plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        #         plt.close('all')
        #     for i in range(predicted_mean_traj_kalman.shape[1] - 1):
        #         name = key + 'Kalman_rollout_phase_portrait' + str(i) + '.pdf'
        #         plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
        #                  label='True')
        #         plt.plot(predicted_mean_traj_kalman[:, i],
        #                  predicted_mean_traj_kalman[:, i + 1],
        #                  label='Predicted', c='b', alpha=0.7)
        #         plt.fill_between(predicted_mean_traj_kalman[:, i],
        #                          predicted_traj_lowconf_kalman[:, i + 1],
        #                          predicted_traj_uppconf_kalman[:, i + 1],
        #                          facecolor='blue', alpha=0.2)
        #         plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
        #                     marker='x', s=100, label='Initial state')
        #         plt.legend()
        #         plt.xlim(xmin=time[0])
        #         plt.xlabel(r'$x_{}$'.format(i + 1))
        #         plt.ylabel(r'$x_{}$'.format(i + 2))
        #         plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        #         plt.close('all')
        #
        #     if plot_output:
        #         # Traj of system output
        #         y_observed = reshape_dim1(
        #             observe_data(torch.as_tensor(true_mean)))
        #         if true_meas_noise_var != 0:
        #             y_observed = reshape_pt1(y_observed + np.random.normal(
        #                 0, np.sqrt(true_meas_noise_var), y_observed.shape))
        #         name = key + 'Kalman_rollout_output_predictions.pdf'
        #         plt.plot(time[100:], y_observed[100:], 'g',
        #                  label='Observed output')
        #         plt.plot(time[100:], observe_data(
        #             torch.as_tensor(predicted_mean_traj_kalman)[100:]),
        #                  label='Estimated output', c='orange', alpha=0.9)
        #         plt.legend()
        #         plt.xlim(xmin=time[0])
        #         plt.xlabel(r'$t$')
        #         plt.ylabel(r'$y$')
        #         plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
        #         plt.close('all')
        # except FileNotFoundError:
        #     print('Files ' + os.path.join(subfolder, str(key)) + ' Kalman not '
        #                                                          'found')

        try:
            # Closed-loop trajs
            name = key + '/Predicted_mean_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_mean_traj_closedloop = data.drop(data.columns[0],
                                                       axis=1).values
            name = key + '/Predicted_uppconf_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_uppconf_closedloop = data.drop(data.columns[0],
                                                          axis=1).values
            name = key + '/Predicted_mean_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_lowconf_closedloop = data.drop(data.columns[0],
                                                          axis=1).values
            for i in range(predicted_mean_traj_closedloop.shape[1]):
                name = key + 'Closedloop_rollout_model_predictions' + str(
                    i) + '.pdf'
                plt.plot(time, true_mean[:, i], 'g', label='True')
                plt.plot(time, predicted_mean_traj_closedloop[:, i],
                         label='Estimated',
                         c='orange', alpha=0.9)
                plt.fill_between(time,
                                 predicted_traj_lowconf_closedloop[:, i],
                                 predicted_traj_uppconf_closedloop[:, i],
                                 facecolor='orange', alpha=0.2)
                plt.legend()
                plt.xlim(xmin=time[0])
                plt.xlabel(r'$t$')
                plt.ylabel(r'$x_{}$'.format(i + 1))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
            for i in range(predicted_mean_traj_closedloop.shape[1] - 1):
                name = key + 'Closedloop_rollout_phase_portrait' + str(
                    i) + '.pdf'
                plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                         label='True')
                plt.plot(predicted_mean_traj_closedloop[:, i],
                         predicted_mean_traj_closedloop[:, i + 1],
                         label='Estimated', c='orange', alpha=0.9)
                plt.fill_between(predicted_mean_traj_closedloop[:, i],
                                 predicted_traj_lowconf_closedloop[:, i + 1],
                                 predicted_traj_uppconf_closedloop[:, i + 1],
                                 facecolor='orange', alpha=0.2)
                plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
                            marker='x', s=100, label='Initial state')
                plt.legend()
                plt.xlabel(r'$x_{}$'.format(i + 1))
                plt.ylabel(r'$x_{}$'.format(i + 2))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')

            if plot_output:
                # Traj of system output
                y_observed = reshape_dim1(
                    observe_data(torch.as_tensor(true_mean)))
                if true_meas_noise_var != 0:
                    y_observed = reshape_pt1(y_observed + np.random.normal(
                        0, np.sqrt(true_meas_noise_var), y_observed.shape))
                name = key + 'Closedloop_rollout_output_predictions.pdf'
                plt.plot(time[100:], y_observed[100:], 'g',
                         label='Observed output')
                plt.plot(time[100:], observe_data(
                    torch.as_tensor(predicted_mean_traj_closedloop)[100:]),
                         label='Estimated output', c='orange', alpha=0.9)
                plt.legend()
                plt.xlim(xmin=time[0])
                plt.xlabel(r'$t$')
                plt.ylabel(r'$y$')
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + 'closedloop '
                                                                 'not found')


def box_plots_onemethod(folder, variables=['RMSE'], avoid0=False,
                        ignore_keys=['Box_plots', 'Ignore', 'Test']):
    # List all folders that contain files (and not only other folders) = all
    # folders containing the actual experiments
    subfolders = []
    subnames = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for subdir in dirnames:
            if any(k in subdir for k in ignore_keys):
                continue  # skip these subfolders
            subpath = os.path.join(dirpath, subdir)
            if len([f for f in os.listdir(subpath) if
                    os.path.isfile(os.path.join(subpath, f)) and not (
                            'DS_Store' in f)]) > 0:
                # subdir contains files: it is a subfolder
                subnames.append(subdir)
                subfolders.append(subpath)
        # stop scanning each branch of dirnames that led to subfolders or
        # that contains certain strings
        dirnames[:] = [name for name in dirnames if not name in subnames and
                       not any(k in name for k in ignore_keys)]

    # Gather all variables to plot in dictionary
    vars = dict.fromkeys(variables)
    for key, val in vars.items():
        vars[key] = []
        for subfolder in subfolders:
            try:
                name = key + '.csv'
                data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                                   header=None)
                values = data.drop(data.columns[0], axis=1).values
                if values[-1, 1] > 50:
                    # Ignore folders with values to large (must be problem)
                    logging.warning('Experiment ' + str(subfolder) +
                                    ' was ignored because of unexpectedly '
                                    'large error values')
                    subfolders.remove(subfolder)
                    continue
                if avoid0:
                    # Avoid first value at sample_idx = 0
                    if 'Data_outside' not in key:
                        values = values[np.logical_not(values[:, 0] == 0)]
                vars[key].append(values)
            except FileNotFoundError:
                print(
                    'Files ' + os.path.join(subfolder, str(key)) + ' not found')

    # Make results folder
    results_folder = os.path.join(folder, 'Box_plots')
    os.makedirs(results_folder, exist_ok=True)  # Will overwrite!
    specs_path = os.path.join(results_folder, 'Specifications.txt')
    shutil.copy(os.path.join(subfolders[0], 'Specifications.txt'),
                results_folder)
    with open(specs_path, 'a') as f:
        print('\n', file=f)
        print('\n', file=f)
        print('\n', file=f)
        print(f'Box plots from {len(subfolders)} experiments', file=f)
        print(f'List of experiments used:', file=f)
        for subfolder in subfolders:
            print(subfolder, file=f)

    # Box plots of each variable
    for key, val in vars.items():
        name = key.replace('/', '_') + '.pdf'  # avoid new dirs
        vars_df = pd.DataFrame.from_dict(vars[key], orient='index').transpose()
        vars_df = vars_df.applymap(lambda x: x[0, 1], na_action='ignore')
        sb.boxplot(data=vars_df)
        plt.xlabel('Methods')
        plt.ylabel(key.replace('_', ' '))
        plt.xticks([], [])
        plt.title(f'Values throughout experiments')
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.close('all')


def box_plots(folder, variables=['RMSE'], methods=['y0T_u0T'],
              avoid0=False, ignore_keys=['Box_plots', 'Ignore', 'Test']):
    # List all folders from each method folder = all folders containing the
    # actual experiments
    subfolders = dict.fromkeys(methods)
    for method in methods:
        if os.path.split(method)[1].startswith('Loop_'):
            subfolders[method] = [os.path.join(folder, method)]
        else:
            methpath = os.path.join(folder, method)
            subfolders[method] = [
                os.path.join(methpath, o) for o in os.listdir(methpath) if
                os.path.isdir(os.path.join(methpath, o)) and not any(
                    k in o for k in ignore_keys)]

    # Gather all variables to plot in dictionary
    vars = dict.fromkeys(variables)
    for key, val in vars.items():
        vars[key] = dict.fromkeys(methods)
        for method in methods:
            vars[key][method] = []
            for subfolder in subfolders[method]:
                try:
                    name = key + '.csv'
                    data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                                       header=None)
                    values = data.drop(data.columns[0], axis=1).values
                    if values.shape[1] == 1:
                        values = np.concatenate(
                            (np.arange(len(values)).reshape(-1, 1),
                             values), axis=1)
                    if values[-1, 1] > 50:
                        # Ignore folders with values to large (must be problem)
                        logging.warning('Experiment ' + str(subfolder) +
                                        ' was ignored because of unexpectedly '
                                        'large error values')
                        subfolders[method].remove(subfolder)
                        continue
                    if avoid0:
                        # Avoid first value at sample_idx = 0
                        if 'Data_outside' not in key:
                            values = values[np.logical_not(values[:, 0] == 0)]
                    vars[key][method].append(values)
                except FileNotFoundError:
                    print(f'File {os.path.join(subfolder, str(key))} not found')

    # Make results folder
    results_folder = os.path.join(folder, 'Box_plots')
    os.makedirs(results_folder, exist_ok=True)  # Will overwrite!
    specs_path = os.path.join(results_folder, 'Specifications.txt')
    shutil.copy(os.path.join(subfolders[methods[0]][0], 'Specifications.txt'),
                results_folder)
    with open(specs_path, 'a') as f:
        print('\n', file=f)
        print('\n', file=f)
        print('\n', file=f)
        print(f'Box plots from {len(methods)} methods and '
              f'{sum([len(subfolders[method]) for method in methods])} '
              f'experiments overall', file=f)
        print(f'List of experiments used:', file=f)
        for method in methods:
            for subfolder in subfolders[method]:
                print(subfolder, file=f)

    # Box plots of each variable
    for key, val in vars.items():
        name = key.replace('/', '_') + '.pdf'  # avoid new dirs
        vars_df = pd.DataFrame.from_dict(vars[key], orient='index').transpose()
        vars_df = vars_df.applymap(lambda x: x[0, 1], na_action='ignore')
        # from operator import itemgetter
        # idx = [1, 3, 4, 5]
        # sb.set_palette(itemgetter(*idx)(sb.color_palette("tab10", 6)))
        sb.boxplot(data=vars_df, whis=50)  # TODO
        # vars_df = pd.DataFrame.from_dict(vars[key], orient='index').transpose()
        # plot = np.array([vars[key][method][-1][-1, -1] for method in methods])
        # if 'Parameter' in key:
        #     plt.plot(plot, 'deepskyblue')
        # else:
        #     plt.plot(plot, 'deepskyblue')
        plt.ylabel(r'RMSE')
        # plt.xlabel('Methods')
        # plt.ylabel(key.replace('_', ' '))
        # plt.xticks([i for i in range(0, len(vars_df.columns))],
        #            [method.replace('_', ' ') for method in vars_df.columns])
        # plt.ylabel('RMSE')
        # plt.xticks([i for i in range(0, len(vars_df.columns))],
        #            ['(a)', '(b)', '(c)', '(d)', '(e)'])
        # plt.xticks([i for i in range(0, len(vars_df.columns))],
        #            [
        #                # r'$t_c=0$',
        #                'direct',
        #                'RNN+',
        #                'KKL',
        #                'KKLu'
        #            ])
        # plt.ylim(0)
        # plt.xticks([i for i in range(0, len(vars_df.columns))],
        #            ['5', '10', '20', '40', '60', '100'])
        # plt.ylim(0, 0.7)
        # plt.xlabel(r'$t_c$')
        plt.xticks([i for i in range(0, len(vars_df.columns))],
                   ['-5', '-4', '-3', '-2', '-1'])
        plt.xlabel(r'$\log(\sigma_\epsilon^2)$')
        plt.ylim(0, 0.5)
        # plt.xticks([i for i in range(0, len(vars_df.columns))],
        #            ['0', '1e-5', '5e-5', '1e-4', '5e-4', '1e-3'])
        # plt.xlabel(r'$\sigma_\epsilon^2$')
        # plt.title(f'Value throughout experiments')
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.close('all')
        # with open(specs_path, 'a') as f:  # TODO
        #     print('\n', file=f)
        #     print('\n', file=f)
        #     print('\n', file=f)
        #     for method in methods:
        #         IQR = np.nanquantile(vars_df[method].values, 0.75) - \
        #               np.nanquantile(vars_df[method].values, 0.25)
        #         print(f'Method {method}, variable {key}: mean '
        #               f'{np.nanmean(vars_df[method].values)}, std '
        #               f'{np.nanstd(vars_df[method].values)}, median '
        #               f'{np.nanquantile(vars_df[method].values, 0.5)}, IQR '
        #               f'{IQR}', file=f)
    # for key, val in vars.items():
    #     name = key.replace('/', '_') + '.pdf'  # avoid new dirs
    #     vars_df = pd.DataFrame.from_dict(vars[key], orient='index').transpose()
    #     vars_df = vars_df.applymap(lambda x: x[0, 1], na_action='ignore')
    #     print(vars_df)
    #     print(vars_df[method])
    #     print(vars_df.columns)
    #     print(vars_df.values)
    #     # sb.boxplot(data=vars_df)#, whis=50)
    #     sb.boxplot(x=vars_df, hue=)
    #     plt.ylabel(key.replace('_', ' '))
    #     plt.xticks([i for i in range(0, len(vars_df.columns))],
    #                ['(a)', '(b)', '(c)', '(d)', '(e)'])
    #     plt.title(f'Value throughout experiments')
    #     plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
    #     plt.close('all')
    #     with open(specs_path, 'a') as f:
    #         print('\n', file=f)
    #         print('\n', file=f)
    #         print('\n', file=f)
    #         for method in methods:
    #             print(f'Method {method}, variable {key}: mean {np.nanmean(vars_df[method].values)}, std {np.nanstd(vars_df[method].values)}', file=f)


if __name__ == '__main__':
    # Collect data from given folder
    folder = str(input('Input folder from which to replot:\n'))

    # # Redo plots of variables of form(time, value)
    # # vars_time = ['Loop_9/closedloop_rollout_RMSE', 'Loop_9/rollout_RMSE',
    # #              'Loop_9/Data_outside_GP/Estimation_RMSE', 'Loop_9/RMSE_time']
    # # replot_vars_time(subfolder=folder, variables=vars_time)
    # vars_time = ['Loop_49/Parameter_errors', 'Loop_49/rollout_RMSE',
    #              'Loop_49/State_estimation_RMSE']
    # replot_vars_time(subfolder=folder, variables=vars_time)

    # Rollouts to replot, from subfolder of one experiment (n dims)
    # rollout_plots = [
    #     'Crossval_Fold_0/Loop_0/Test/Test_rollouts_-1/Rollout_0/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_-1/Rollout_1/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_-1/Rollout_2/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_-1/Rollout_3/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_0/Rollout_0/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_0/Rollout_1/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_0/Rollout_2/',
    #                  'Crossval_Fold_0/Loop_0/Test/Test_rollouts_0/Rollout_3/',
    #                  'Crossval_Fold_1/Loop_0/Test/Test_rollouts_1/Rollout_0/',
    #                  'Crossval_Fold_1/Loop_0/Test/Test_rollouts_1/Rollout_1/',
    #                  'Crossval_Fold_1/Loop_0/Test/Test_rollouts_1/Rollout_2/',
    #                  'Crossval_Fold_1/Loop_0/Test/Test_rollouts_1/Rollout_3/',
    #                  ]
    # observation_matrix = torch.tensor([6.92140217e-03, -2.80060136e-03,
    #                                             -1.21308401e-02,
    #                                             -2.64675209e-05,
    #                                             5.56641843e-03])
    # observe_data = lambda x: reshape_dim1(torch.matmul(
    #     reshape_pt1(x), observation_matrix.t()))
    # replot_rollouts(subfolder=folder, plots=rollout_plots,
    #                 plot_output=True, observe_data=observe_data,
    #                 true_meas_noise_var=0.00255366)
    rollout_plots = [
        # 'Loop_0/Rollouts_0/Rollout_17/', 'Loop_9/Rollouts_9/Rollout_17/',
        # 'Loop_0/Rollouts_0/Rollout_19/', 'Loop_9/Rollouts_9/Rollout_19/',
        'Rollouts_whole/Rollout_0/',
    ]
    replot_rollouts(subfolder=folder, plots=rollout_plots,
                    plot_output=True, observe_data=dim14_observe_data)

    # # Variables to box plot, from folder containing several experiments,
    # # of form (time, value)
    # vars_time = ['Loop_9/State_estimation_RMSE',
    #              'Loop_9/Parameter_errors']
    # box_plots_vars_time(folder=folder, variables=vars_time, avoid0=False)

    # # Variables to box plot of form (value) not time-dependent, from folder
    # # containing several experiments
    # vars = ['rollout_RMSE', 'rollout_RMSE_init',
    #         'rollout_RMSE_output'
    #         ]
    # # methods = [
    # #     # 'y0_u0',
    # #     'y0T_u0T',
    # #     'y0T_u0T_RNN_outNN_back',
    # #     'KKL_u0T_back',
    # #     'KKLu_back'
    # # ]
    # methods = [
    #     # 'optimD1_tc5/y0T_u0T_RNN_outNN_back',
    #     #        'optimD1_tc10/y0T_u0T_RNN_outNN_back',
    #     #        'optimD1_tc20/y0T_u0T_RNN_outNN_back',
    #     #        'optimD1_tc40/y0T_u0T_RNN_outNN_back',
    #     #        'optimD1_tc60/y0T_u0T_RNN_outNN_back',
    #     #        'optimD1_tc100/y0T_u0T_RNN_outNN_back',
    #     'optimD2_noisevar1e-5/KKL_u0T_back',
    #     'optimD2_noisevar1e-4/KKL_u0T_back',
    #     'optimD2_noisevar1e-3/KKL_u0T_back',
    #     'optimD2_noisevar1e-2/KKL_u0T_back',
    #     'optimD2_noisevar1e-1/KKL_u0T_back',
    #            ]
    # # vars = ['rollout_RMSE', 'rollout_RMSE_output', 'rollout_RMSE_init']
    # # methods = [
    # #             'No_struct/100_rollouts/y0T_u0T',
    # #             'Hamiltonian/100_rollouts/y0T_u0T',
    # #             'Hamiltonian_x1dot=x2/100_rollouts/y0T_u0T',
    # #             'Paramid_lin_observer/100_rollouts/y0T_u0T',
    # #             'Extended_model_only_recog/100_rollouts/y0T_u0T']
    # # vars = ['State_estimation_RMSE', 'Parameter_errors']
    # # methods = [
    # #             'Manuscript1_noise0/Loop_9',
    # #             'Manuscript1_noise1e-5/Loop_9',
    # #             'Manuscript1_noise5e-5/Loop_9',
    # #             'Manuscript1_noise1e-4/Loop_9',
    # #             'Manuscript1_noise5e-4/Loop_9',
    # #             'Manuscript1_noise1e-3/Loop_9',
    # # ]
    # ignore_keys = ['Box_plots', 'Ignore', 'Test', 'test', 'Wrong', 'wrong',
    #                'bad', 'Bad']
    # # box_plots_onemethod(folder=folder, variables=vars, avoid0=False)
    # box_plots(folder=folder, variables=vars, methods=methods, avoid0=False,
    #           ignore_keys=ignore_keys)
