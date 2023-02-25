import logging
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

from model_evaluation.plotting_closedloop_rollouts import \
    save_closedloop_rollout_variables, plot_test_closedloop_rollout_data, \
    plot_val_closedloop_rollout_data
from model_evaluation.plotting_functions import plot_model_evaluation, \
    run_rollouts
from model_evaluation.plotting_kalman_rollouts import \
    save_kalman_rollout_variables, plot_val_kalman_rollout_data, \
    plot_test_kalman_rollout_data
from model_evaluation.plotting_rollouts import plot_val_rollout_data, \
    plot_test_rollout_data, save_rollout_variables
from utils.config import Config
from utils.utils import remove_outlier, reshape_pt1, concatenate_lists, save_log
from .simple_GPyTorch_dyn import Simple_GPyTorch_Dyn

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

# Class to learn simple dynamics GP from several experimental datasets (hence
# usually ground truth is only approximated)
# Inherits from Simple_GP_Dyn, basically the same but ground_truth_approx =
# True by default and model evaluation tools (evaluation grid, rollouts,
# GP plot) are chosen close to training data and updated at each new learning
# loop to incorporate new training data


LOCAL_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
LOCAL_PATH_TO_SRC = LOCAL_PATH.split(os.sep + 'src', 1)[0]


class GP_Dyn_Several_Exp_Datasets(Simple_GPyTorch_Dyn):

    def __init__(self, X, U, Y, config: Config):
        super().__init__(X=X, U=U, Y=Y, config=config, ground_truth_approx=True)

        if self.nb_rollouts > 1 and self.ground_truth_approx:
            logging.warning('No use in having a high number of rollouts when '
                            'no ground truth is available, since the real '
                            'model evaluation is obtained by predicting on a '
                            'val set, not on computing rollout RMSE or any '
                            'other type of metric using the ground truth!')
        self.test_RMSE = torch.zeros((0, 2))
        self.test_SRMSE = torch.zeros((0, 2))
        self.test_log_AL = torch.zeros((0, 2))
        self.test_stand_log_AL = torch.zeros((0, 2))
        self.val_RMSE = torch.zeros((0, 2))
        self.val_SRMSE = torch.zeros((0, 2))
        self.val_log_AL = torch.zeros((0, 2))
        self.val_stand_log_AL = torch.zeros((0, 2))
        self.test_rollout_RMSE = torch.zeros((0, 2))
        self.test_rollout_SRMSE = torch.zeros((0, 2))
        self.test_rollout_log_AL = torch.zeros((0, 2))
        self.test_rollout_stand_log_AL = torch.zeros((0, 2))
        self.val_rollout_RMSE = torch.zeros((0, 2))
        self.val_rollout_SRMSE = torch.zeros((0, 2))
        self.val_rollout_log_AL = torch.zeros((0, 2))
        self.val_rollout_stand_log_AL = torch.zeros((0, 2))
        self.observer_test_RMSE = torch.zeros((0, 2))
        self.observer_test_SRMSE = torch.zeros((0, 2))
        self.observer_val_RMSE = torch.zeros((0, 2))
        self.observer_val_SRMSE = torch.zeros((0, 2))
        self.output_test_RMSE = torch.zeros((0, 2))
        self.output_test_SRMSE = torch.zeros((0, 2))
        self.output_val_RMSE = torch.zeros((0, 2))
        self.output_val_SRMSE = torch.zeros((0, 2))
        self.test_closedloop_rollout_RMSE = torch.zeros((0, 2))
        self.test_closedloop_rollout_SRMSE = torch.zeros((0, 2))
        self.test_closedloop_rollout_log_AL = torch.zeros((0, 2))
        self.test_closedloop_rollout_stand_log_AL = torch.zeros((0, 2))
        self.val_closedloop_rollout_RMSE = torch.zeros((0, 2))
        self.val_closedloop_rollout_SRMSE = torch.zeros((0, 2))
        self.val_closedloop_rollout_log_AL = torch.zeros((0, 2))
        self.val_closedloop_rollout_stand_log_AL = torch.zeros((0, 2))
        self.test_kalman_rollout_RMSE = torch.zeros((0, 2))
        self.test_kalman_rollout_SRMSE = torch.zeros((0, 2))
        self.test_kalman_rollout_log_AL = torch.zeros((0, 2))
        self.test_kalman_rollout_stand_log_AL = torch.zeros((0, 2))
        self.val_kalman_rollout_RMSE = torch.zeros((0, 2))
        self.val_kalman_rollout_SRMSE = torch.zeros((0, 2))
        self.val_kalman_rollout_log_AL = torch.zeros((0, 2))
        self.val_kalman_rollout_stand_log_AL = torch.zeros((0, 2))

        if self.__class__.__name__ == 'GP_Dyn_Several_Exp_Datasets':
            # Only do if constructor not called from inherited class
            if not self.existing_results_folder:
                self.results_folder = \
                    self.results_folder.replace('_pass', '_fold_crossval')
                self.results_folder = self.results_folder.replace('/Loop_0', '')
            else:
                self.results_folder = self.existing_results_folder
            self.results_folder = os.path.join(self.results_folder,
                                               'Crossval_Fold_' + str(
                                                   self.fold_nb))
            if self.save_inside_fold:
                self.results_folder = os.path.join(self.results_folder,
                                                   'Loop_0')
            os.makedirs(self.results_folder, exist_ok=False)
            self.validation_folder = os.path.join(self.results_folder,
                                                  'Validation')
            os.makedirs(self.validation_folder, exist_ok=True)
            self.test_folder = os.path.join(self.results_folder, 'Test')
            os.makedirs(self.test_folder, exist_ok=True)
            self.save_grid_variables(self.grid, self.grid_controls,
                                     self.true_predicted_grid,
                                     self.results_folder)
            self.true_predicted_grid = torch.as_tensor(
                self.true_predicted_grid, device=self.device)
            self.grid = torch.as_tensor(self.grid, device=self.device)
            self.grid_controls = torch.as_tensor(self.grid_controls,
                                                 device=self.device)
            self.grid_variables = {'Evaluation_grid': self.grid,
                                   'Grid_controls': self.grid_controls}
            save_rollout_variables(self, self.results_folder, self.nb_rollouts,
                                   self.rollout_list, step=self.step,
                                   ground_truth_approx=self.ground_truth_approx,
                                   plots=self.monitor_experiment)
            # Save log in results folder
            if not self.existing_results_folder:
                os.rename(
                    str(LOCAL_PATH_TO_SRC) + '/Figures/Logs/' + 'log' +
                    str(sys.argv[1]) + '.log',
                    os.path.join(self.results_folder,
                                 'log' + str(sys.argv[1]) + '.log'))
            save_log(self.results_folder)
            if self.verbose:
                logging.info(self.results_folder)

    def update_data(self, new_X=[], new_U=[], new_Y=[]):
        whole_X, whole_U, whole_Y = Simple_GPyTorch_Dyn.update_data(
            self, new_X=new_X, new_U=new_U, new_Y=new_Y)

        # Recreate and resave evaluation grid and list of rollouts with
        # updated training data
        if (len(new_X) > 0) and (len(new_U) > 0) and (len(new_Y) > 0):
            old_X = self.X
            old_U = self.U
            old_Y = self.Y
            self.X = whole_X
            self.U = whole_U
            self.Y = whole_Y
            self.grid, self.grid_controls = self.create_grid(self.init_state,
                                                             self.init_control,
                                                             self.constrain_u,
                                                             self.grid_inf,
                                                             self.grid_sup)
            true_predicted_grid = \
                self.create_true_predicted_grid(self.grid, self.grid_controls)
            self.true_predicted_grid = \
                true_predicted_grid.reshape(-1, self.Y.shape[1])
            # Reject outliers from grid
            true_predicted_grid_df = pd.DataFrame(
                self.true_predicted_grid.cpu().numpy())
            grid_df = pd.DataFrame(self.grid.cpu().numpy())
            grid_controls_df = pd.DataFrame(self.grid_controls.cpu().numpy())
            mask = remove_outlier(true_predicted_grid_df)
            true_predicted_grid_df = true_predicted_grid_df[mask]
            grid_df = grid_df[mask]
            grid_controls_df = grid_controls_df[mask]
            self.true_predicted_grid = \
                true_predicted_grid_df.values.reshape(-1, self.Y.shape[1])
            self.grid = grid_df.values.reshape(-1, self.X.shape[1])
            self.grid_controls = \
                grid_controls_df.values.reshape(-1, self.U.shape[1])
            self.save_grid_variables(self.grid, self.grid_controls,
                                     self.true_predicted_grid,
                                     self.results_folder)
            self.true_predicted_grid = torch.as_tensor(
                self.true_predicted_grid, device=self.device)
            self.grid = torch.as_tensor(self.grid, device=self.device)
            self.grid_controls = torch.as_tensor(self.grid_controls,
                                                 device=self.device)
            self.grid_variables = {'Evaluation_grid': self.grid,
                                   'Grid_controls': self.grid_controls}
            self.rollout_list = self.create_rollout_list()
            save_rollout_variables(self, self.results_folder, self.nb_rollouts,
                                   self.rollout_list, step=self.step - 1,
                                   ground_truth_approx=self.ground_truth_approx,
                                   plots=self.monitor_experiment)
            self.X = old_X
            self.U = old_U
            self.Y = old_Y
            del whole_X, whole_U, whole_Y

    def create_test_rollouts(self, X_test, U_test, Y_test, cut_idx):
        test_rollout_list = []
        for i in range(len(cut_idx)):
            start_idx = cut_idx[i]  # start of test traj
            if i < len(cut_idx) - 1:
                end_idx = cut_idx[i + 1]  # end of test traj (excluded)
            else:
                end_idx = len(Y_test)
            if self.full_rollouts:
                init_state = reshape_pt1(X_test[start_idx])
                true_mean = torch.cat((
                    init_state, reshape_pt1(Y_test[start_idx:end_idx])),
                    dim=0)
                control_traj = reshape_pt1(U_test[start_idx:end_idx])
                test_rollout_list.append(
                    [init_state, control_traj, true_mean])
            else:
                nb_local_rollouts = int(
                    np.floor((torch.tensor(end_idx - start_idx) /
                              self.rollout_length)))
                for j in range(nb_local_rollouts):
                    random_start_idx = np.random.randint(
                        start_idx, end_idx - self.rollout_length)
                    random_end_idx = random_start_idx + self.rollout_length
                    init_state = reshape_pt1(X_test[random_start_idx])
                    # true_mean = reshape_pt1(
                    #     X_test[random_start_idx:random_end_idx + 1])
                    true_mean = torch.cat((
                        init_state, reshape_pt1(
                            Y_test[random_start_idx:random_end_idx])),
                        dim=0)
                    control_traj = reshape_pt1(
                        U_test[random_start_idx:random_end_idx])
                    test_rollout_list.append(
                        [init_state, control_traj, true_mean])
        self.test_rollout_list = test_rollout_list
        self.nb_test_rollouts = len(self.test_rollout_list)

    def evaluate_test_rollouts(self, only_prior=False):
        if len(self.test_rollout_list) == 0:
            return 0

        # save_rollout_variables(self, self.test_folder, len(self.test_rollout_list),
        #                        self.test_rollout_list, step=self.step - 1,
        #                        results=False,
        #                        ground_truth_approx=self.ground_truth_approx,
        #                        title='Test_rollouts',
        #                        plots=self.monitor_experiment)
        # folder = os.path.join(self.test_folder,
        #                       'Test_rollouts' + '_' + str(self.step - 1))
        # os.makedirs(folder, exist_ok=True)  # TODO enough?
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.test_rollout_list,
                         folder=self.test_folder, only_prior=only_prior)
        self.specs['test_rollout_RMSE'] = rollout_RMSE
        self.specs['test_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_rollout_log_AL'] = rollout_log_AL
        self.specs['test_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.test_rollout_RMSE = \
            torch.cat((self.test_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_RMSE]))), dim=0)
        self.test_rollout_SRMSE = \
            torch.cat((self.test_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_SRMSE]))),
                      dim=0)
        self.test_rollout_log_AL = \
            torch.cat((self.test_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.test_rollout_stand_log_AL = \
            torch.cat((self.test_rollout_stand_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_stand_log_AL]))), dim=0)
        self.variables['test_rollout_RMSE'] = self.test_rollout_RMSE
        self.variables['test_rollout_SRMSE'] = self.test_rollout_SRMSE
        self.variables['test_rollout_log_AL'] = self.test_rollout_log_AL
        self.variables['test_rollout_stand_log_AL'] = \
            self.test_rollout_stand_log_AL
        plot_test_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = concatenate_lists(self.test_rollout_list,
                                                       rollout_list)
        save_rollout_variables(
            self, self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1, results=True,
            ground_truth_approx=self.ground_truth_approx,
            title='Test_rollouts', plots=self.monitor_experiment)

    def evaluate_test_kalman_rollouts(self, observer, observe_data,
                                      discrete_observer,
                                      no_GP_in_observer=False,
                                      only_prior=False):
        if len(self.test_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, self.test_rollout_list, folder=self.test_folder,
            observer=observer, observe_data=observe_data,
            discrete_observer=discrete_observer, kalman=True,
            no_GP_in_observer=no_GP_in_observer, only_prior=only_prior)
        self.specs['test_kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['test_kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['test_kalman_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.test_kalman_rollout_RMSE = \
            torch.cat((self.test_kalman_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.test_kalman_rollout_SRMSE = \
            torch.cat((self.test_kalman_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.test_kalman_rollout_log_AL = \
            torch.cat((self.test_kalman_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.test_kalman_rollout_stand_log_AL = torch.cat((
            self.test_kalman_rollout_stand_log_AL, reshape_pt1(torch.tensor([
                torch.tensor(self.sample_idx), rollout_stand_log_AL]))), dim=0)
        self.variables['test_kalman_rollout_RMSE'] = \
            self.test_kalman_rollout_RMSE
        self.variables['test_kalman_rollout_SRMSE'] = \
            self.test_kalman_rollout_SRMSE
        self.variables['test_kalman_rollout_log_AL'] = \
            self.test_kalman_rollout_log_AL
        self.variables['test_kalman_rollout_stand_log_AL'] = \
            self.test_kalman_rollout_stand_log_AL
        plot_test_kalman_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = concatenate_lists(self.test_rollout_list,
                                                       rollout_list)
        save_kalman_rollout_variables(
            self, self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx,
            title='Test_rollouts', plots=self.monitor_experiment)

    def evaluate_test_closedloop_rollouts(self, observer, observe_data,
                                          no_GP_in_observer=False):
        if len(self.test_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, self.test_rollout_list, folder=self.test_folder,
            observer=observer, observe_data=observe_data, closedloop=True,
            no_GP_in_observer=no_GP_in_observer)
        self.specs['test_closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['test_closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['test_closedloop_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.test_closedloop_rollout_RMSE = \
            torch.cat((self.test_closedloop_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.test_closedloop_rollout_SRMSE = \
            torch.cat((self.test_closedloop_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.test_closedloop_rollout_log_AL = \
            torch.cat((self.test_closedloop_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.test_closedloop_rollout_stand_log_AL = torch.cat((
            self.test_closedloop_rollout_stand_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_stand_log_AL]))), dim=0)
        self.variables['test_closedloop_rollout_RMSE'] = \
            self.test_closedloop_rollout_RMSE
        self.variables['test_closedloop_rollout_SRMSE'] = \
            self.test_closedloop_rollout_SRMSE
        self.variables['test_closedloop_rollout_log_AL'] = \
            self.test_closedloop_rollout_log_AL
        self.variables['test_closedloop_rollout_stand_log_AL'] = \
            self.test_closedloop_rollout_stand_log_AL
        plot_test_closedloop_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = concatenate_lists(self.test_rollout_list,
                                                       rollout_list)
        save_closedloop_rollout_variables(
            self, self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx,
            title='Test_rollouts', plots=self.monitor_experiment)

    def create_val_rollouts(self, X_val, U_val, Y_val, cut_idx):
        val_rollout_list = []
        for i in range(len(cut_idx)):
            start_idx = cut_idx[i]  # start of val traj
            if i < len(cut_idx) - 1:
                end_idx = cut_idx[i + 1]  # end of val traj (excluded)
            else:
                end_idx = len(Y_val)
            if self.full_rollouts:
                init_state = reshape_pt1(X_val[start_idx])
                true_mean = reshape_pt1(X_val[start_idx:end_idx + 1])
                control_traj = reshape_pt1(U_val[start_idx:end_idx])
                val_rollout_list.append(
                    [init_state, control_traj, true_mean])
            else:
                nb_local_rollouts = int(
                    np.floor((end_idx - start_idx) / self.rollout_length))
                for j in range(nb_local_rollouts):
                    random_start_idx = np.random.randint(
                        start_idx, end_idx - self.rollout_length)
                    random_end_idx = random_start_idx + self.rollout_length
                    init_state = reshape_pt1(X_val[random_start_idx])
                    true_mean = reshape_pt1(
                        X_val[random_start_idx:random_end_idx + 1])
                    control_traj = reshape_pt1(
                        U_val[random_start_idx:random_end_idx])
                    val_rollout_list.append(
                        [init_state, control_traj, true_mean])
        self.val_rollout_list = val_rollout_list
        self.nb_val_rollouts = len(self.val_rollout_list)

    def evaluate_val_rollouts(self, only_prior=False):
        if len(self.val_rollout_list) == 0:
            return 0

        # save_rollout_variables(self, self.val_folder, len(self.val_rollout_list),
        #                        self.val_rollout_list, step=self.step - 1,
        #                        results=False,
        #                        ground_truth_approx=self.ground_truth_approx,
        #                        title='Val_rollouts',
        #                        plots=self.monitor_experiment)
        # folder = os.path.join(self.test_folder,
        #                       'Val_rollouts' + '_' + str(self.step - 1))
        # os.makedirs(folder, exist_ok=True)  # TODO enough?
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.val_rollout_list,
                         folder=self.val_folder, only_prior=only_prior)
        self.specs['val_rollout_RMSE'] = rollout_RMSE
        self.specs['val_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_rollout_log_AL'] = rollout_log_AL
        self.specs['val_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.val_rollout_RMSE = \
            torch.cat((self.val_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_RMSE]))), dim=0)
        self.val_rollout_SRMSE = \
            torch.cat((self.val_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_SRMSE]))),
                      dim=0)
        self.val_rollout_log_AL = \
            torch.cat((self.val_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.val_rollout_stand_log_AL = \
            torch.cat((self.val_rollout_stand_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_stand_log_AL]))), dim=0)
        self.variables['val_rollout_RMSE'] = self.val_rollout_RMSE
        self.variables['val_rollout_SRMSE'] = self.val_rollout_SRMSE
        self.variables['val_rollout_log_AL'] = self.val_rollout_log_AL
        self.variables['val_rollout_stand_log_AL'] = \
            self.val_rollout_stand_log_AL
        plot_val_rollout_data(self, folder=self.val_folder)
        complete_val_rollout_list = concatenate_lists(self.val_rollout_list,
                                                      rollout_list)
        save_rollout_variables(
            self, self.val_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1, results=True,
            ground_truth_approx=self.ground_truth_approx, title='Val_rollouts',
            plots=self.monitor_experiment)

    def evaluate_val_kalman_rollouts(self, observer, observe_data,
                                     discrete_observer,
                                     no_GP_in_observer=False,
                                     only_prior=False):
        if len(self.val_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, self.val_rollout_list, folder=self.val_folder,
            observer=observer, observe_data=observe_data,
            discrete_observer=discrete_observer, kalman=True,
            no_GP_in_observer=no_GP_in_observer, only_prior=only_prior)
        self.specs['val_kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['val_kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['val_kalman_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.val_kalman_rollout_RMSE = \
            torch.cat((self.val_kalman_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.val_kalman_rollout_SRMSE = \
            torch.cat((self.val_kalman_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.val_kalman_rollout_log_AL = \
            torch.cat((self.val_kalman_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.val_kalman_rollout_stand_log_AL = torch.cat((
            self.val_kalman_rollout_stand_log_AL, reshape_pt1(torch.tensor([
                torch.tensor(self.sample_idx), rollout_stand_log_AL]))), dim=0)
        self.variables['val_kalman_rollout_RMSE'] = \
            self.val_kalman_rollout_RMSE
        self.variables['val_kalman_rollout_SRMSE'] = \
            self.val_kalman_rollout_SRMSE
        self.variables['val_kalman_rollout_log_AL'] = \
            self.val_kalman_rollout_log_AL
        self.variables['val_kalman_rollout_stand_log_AL'] = \
            self.val_kalman_rollout_stand_log_AL
        plot_val_kalman_rollout_data(self, folder=self.val_folder)
        complete_val_rollout_list = concatenate_lists(self.val_rollout_list,
                                                      rollout_list)
        save_kalman_rollout_variables(
            self, self.val_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx,
            title='Val_rollouts', plots=self.monitor_experiment)

    def evaluate_val_closedloop_rollouts(self, observer, observe_data,
                                         no_GP_in_observer=False):
        if len(self.val_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, self.val_rollout_list, folder=self.val_folder,
            observer=observer, observe_data=observe_data, closedloop=True,
            no_GP_in_observer=no_GP_in_observer)
        self.specs['val_closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['val_closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['val_closedloop_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.val_closedloop_rollout_RMSE = \
            torch.cat((self.val_closedloop_rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.val_closedloop_rollout_SRMSE = \
            torch.cat((self.val_closedloop_rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.val_closedloop_rollout_log_AL = \
            torch.cat((self.val_closedloop_rollout_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_log_AL]))), dim=0)
        self.val_closedloop_rollout_stand_log_AL = torch.cat((
            self.val_closedloop_rollout_stand_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_stand_log_AL]))), dim=0)
        self.variables['val_closedloop_rollout_RMSE'] = \
            self.val_closedloop_rollout_RMSE
        self.variables['val_closedloop_rollout_SRMSE'] = \
            self.val_closedloop_rollout_SRMSE
        self.variables['val_closedloop_rollout_log_AL'] = \
            self.val_closedloop_rollout_log_AL
        self.variables['val_closedloop_rollout_stand_log_AL'] = \
            self.val_closedloop_rollout_stand_log_AL
        plot_val_closedloop_rollout_data(self, folder=self.val_folder)
        complete_val_rollout_list = concatenate_lists(self.val_rollout_list,
                                                      rollout_list)
        save_closedloop_rollout_variables(
            self, self.val_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx,
            title='Val_rollouts', plots=self.monitor_experiment)

    def test(self, X_test, U_test, Y_test, cut_idx=[]):
        # Save test data
        if len(torch.nonzero(X_test, as_tuple=False)) == 0:
            self.test_rollout_list = []
            self.nb_test_rollouts = 0
            return 0

        self.test_folder = os.path.join(self.results_folder, 'Test')
        os.makedirs(self.test_folder, exist_ok=True)

        if self.monitor_experiment:
            name = 'X_test'
            for i in range(X_test.shape[1]):
                plt.plot(X_test[:, i], label='Test input')
                plt.title('Input data used for testing')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Input')
                plt.savefig(os.path.join(self.test_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(X_test.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            name = 'U_test'
            for i in range(U_test.shape[1]):
                plt.plot(U_test[:, i], label='Test control')
                plt.title('Control data used for testing')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Control')
                plt.savefig(os.path.join(self.test_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(U_test.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            name = 'Y_test'
            for i in range(Y_test.shape[1]):
                plt.plot(Y_test[:, i], label='Test output')
                plt.title('Output data used for testing')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Output')
                plt.savefig(os.path.join(self.test_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(Y_test.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

        # Put test data on GPU
        if torch.cuda.is_available():
            X_test, U_test, Y_test = X_test.cuda(), U_test.cuda(), Y_test.cuda()

        if self.monitor_experiment:
            # Evaluate model
            nb_points = int(np.ceil(np.min([len(X_test), 1000])))
            self.test_grid_random_idx = torch.randperm(len(X_test))[:nb_points]
            X_test_grid = X_test[self.test_grid_random_idx]
            U_test_grid = U_test[self.test_grid_random_idx]
            Y_test_grid = Y_test[self.test_grid_random_idx]
            if 'Michelangelo' in self.system:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_test_grid, grid_controls=U_test_grid,
                        true_predicted_grid=Y_test_grid,
                        use_euler='Michelangelo')
            elif ('justvelocity' in self.system) and not self.continuous_model:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_test_grid, grid_controls=U_test_grid,
                        true_predicted_grid=Y_test_grid,
                        use_euler='discrete_justvelocity')
            elif ('justvelocity' in self.system) and self.continuous_model:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_test_grid, grid_controls=U_test_grid,
                        true_predicted_grid=Y_test_grid,
                        use_euler='continuous_justvelocity')
            else:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_test_grid, grid_controls=U_test_grid,
                        true_predicted_grid=Y_test_grid, use_euler=None)

            # Update all evaluation variables
            self.test_RMSE = torch.cat((self.test_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), RMSE]))), dim=0)
            self.test_SRMSE = torch.cat((self.test_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), SRMSE]))), dim=0)
            self.test_log_AL = torch.cat((self.test_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), log_likelihood]))),
                                         dim=0)
            self.test_stand_log_AL = torch.cat((
                self.test_stand_log_AL, reshape_pt1(torch.tensor([torch.tensor(
                    self.sample_idx), stand_log_likelihood]))), dim=0)

            # Save plot and csv files of test_RMSE and test_log_AL
            name = 'Test_RMSE'
            plt.plot(self.test_RMSE[:, 0] - 1, self.test_RMSE[:, 1], 'c',
                     label='RMSE')
            plt.title('RMSE between model and true dynamics over test data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('RMSE')
            plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.test_RMSE.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            name = 'Test_SRMSE'
            plt.plot(self.test_SRMSE[:, 0] - 1, self.test_SRMSE[:, 1], 'c',
                     label='SRMSE')
            plt.title('Standardized RMSE between model and true dynamics over '
                      'test data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('SRMSE')
            plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.test_SRMSE.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            name = 'Test_average_log_likelihood'
            plt.plot(self.test_log_AL[:, 0], self.test_log_AL[:, 1], 'c',
                     label='log_AL')
            plt.title('Average log likelihood between model and true dynamics '
                      'over test data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Average log likelihood')
            plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.test_log_AL.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            name = 'Test_standardized_average_log_likelihood'
            plt.plot(self.test_stand_log_AL[:, 0], self.test_stand_log_AL[:, 1],
                     'c', label='stand_log_AL')
            plt.title('Standardized average log likelihood between model and '
                      'true  dynamics over test data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Standardized average log likelihood')
            plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.test_stand_log_AL.numpy())
            file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                        header=False)

            # Plot model evaluation scatter plot over the test data
            plot_model_evaluation(Evaluation_grid=X_test_grid,
                                  Grid_controls=U_test_grid,
                                  Predicted_grid=predicted_grid,
                                  True_predicted_grid=Y_test_grid,
                                  folder=self.test_folder,
                                  ground_truth_approx=self.ground_truth_approx,
                                  title='Test_model_evaluation', verbose=False)
            # self.save_grid_variables(X_test_grid, U_test_grid, Y_test_grid,
            #                          results_folder=self.test_folder)
            filename = 'Predicted_Ytest' + str(self.step) + '.csv'
            file = pd.DataFrame(predicted_grid.cpu().numpy())
            file.to_csv(os.path.join(self.test_folder, filename), header=False)

        # Rollouts over the test data: either random subsets of test data,
        # number proportional to length of test data, or full test scenario
        # for each rollout
        if (self.fold_nb == 0) and (self.step == 1):
            self.create_test_rollouts(X_test, U_test, Y_test, cut_idx)
            save_rollout_variables(self, self.test_folder,
                                   len(self.test_rollout_list),
                                   self.test_rollout_list, step=self.step - 1,
                                   results=False,
                                   ground_truth_approx=self.ground_truth_approx,
                                   title='Test_rollouts',
                                   plots=self.monitor_experiment)  # TODO
        elif (self.fold_nb > 0) and (self.step == 1):
            previous_test_folder = self.test_folder.replace(
                'Crossval_Fold_' + str(self.fold_nb), 'Crossval_Fold_0')
            path, dirs, files = next(os.walk(os.path.join(
                previous_test_folder, 'Test_rollouts0')))
            self.nb_test_rollouts = len(dirs)
            self.test_rollout_list = self.read_rollout_list(
                # TODO save before to GPU?
                previous_test_folder, self.nb_test_rollouts, step=self.step - 1,
                folder_title='Test_rollouts', save=self.test_folder)
            # save_rollout_variables(self, self.test_folder,
            #                        len(self.test_rollout_list),
            #                        self.test_rollout_list, step=self.step - 1,
            #                        results=False,
            #                        ground_truth_approx=self.ground_truth_approx,
            #                        title='Test_rollouts',
            #                        plots=self.monitor_experiment)
        self.evaluate_test_rollouts()

    def validate(self, X_val, U_val, Y_val, cut_idx=[]):
        # Save validation data
        if len(torch.nonzero(X_val, as_tuple=False)) == 0:
            self.val_rollout_list = []
            self.nb_val_rollouts = 0
            return 0

        # Read and save validation data
        self.validation_folder = os.path.join(self.results_folder, 'Validation')
        os.makedirs(self.validation_folder, exist_ok=True)

        if self.monitor_experiment:
            name = 'X_val'
            for i in range(X_val.shape[1]):
                plt.plot(X_val[:, i], label='Validation input')
                plt.title('Input data used for validation')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Input')
                plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(X_val.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            name = 'U_val'
            for i in range(U_val.shape[1]):
                plt.plot(U_val[:, i], label='Validation control')
                plt.title('Control data used for validation')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Control')
                plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(U_val.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            name = 'Y_val'
            for i in range(Y_val.shape[1]):
                plt.plot(Y_val[:, i], label='Validation output')
                plt.title('Output data used for validation')
                plt.legend()
                plt.xlabel('Number of samples')
                plt.ylabel('Output')
                plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                         '.pdf'), bbox_inches='tight')
                plt.close('all')
            file = pd.DataFrame(Y_val.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

        # Put validation data on GPU
        if torch.cuda.is_available():
            X_val, U_val, Y_val = X_val.cuda(), U_val.cuda(), Y_val.cuda()

        if self.monitor_experiment:
            # Evaluate model
            nb_points = int(np.ceil(np.min([len(X_val), 1000])))
            self.test_grid_random_idx = torch.randperm(len(X_val))[:nb_points]
            X_val_grid = X_val[self.val_grid_random_idx]
            U_val_grid = U_val[self.val_grid_random_idx]
            Y_val_grid = Y_val[self.val_grid_random_idx]
            if 'Michelangelo' in self.system:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_val, grid_controls=U_val,
                        true_predicted_grid=Y_val,
                        use_euler='Michelangelo')
            elif ('justvelocity' in self.system) and not self.continuous_model:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_val_grid, grid_controls=U_val_grid,
                        true_predicted_grid=Y_val_grid,
                        use_euler='discrete_justvelocity')
            elif ('justvelocity' in self.system) and self.continuous_model:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_val_grid, grid_controls=U_val_grid,
                        true_predicted_grid=Y_val_grid,
                        use_euler='continuous_justvelocity')
            else:
                RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
                grid_controls, log_likelihood, stand_log_likelihood = \
                    self.compute_l2error_grid(
                        grid=X_val_grid, grid_controls=U_val_grid,
                        true_predicted_grid=Y_val_grid,
                        use_euler=None)

            # Update all evaluation variables
            self.val_RMSE = torch.cat((self.val_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), RMSE]))), dim=0)
            self.val_SRMSE = torch.cat((self.val_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), SRMSE]))), dim=0)
            self.val_log_AL = torch.cat((self.val_log_AL, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), log_likelihood]))),
                                        dim=0)
            self.val_stand_log_AL = \
                torch.cat((self.val_stand_log_AL, reshape_pt1(torch.tensor([
                    torch.tensor(self.sample_idx), stand_log_likelihood]))),
                          dim=0)

            # Save plot and csv files of val_RMSE and val_log_AL
            name = 'Validation_RMSE'
            plt.plot(self.val_RMSE[:, 0] - 1, self.val_RMSE[:, 1], 'c',
                     label='RMSE')
            plt.title(
                'RMSE between model and true dynamics over validation data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('RMSE')
            plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.val_RMSE.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            name = 'Validation_SRMSE'
            plt.plot(self.val_SRMSE[:, 0] - 1, self.val_SRMSE[:, 1], 'c',
                     label='SRMSE')
            plt.title('Standardized RMSE between model and true dynamics over '
                      'validation data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('SRMSE')
            plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.val_SRMSE.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            name = 'Validation_average_log_likelihood'
            plt.plot(self.val_log_AL[:, 0], self.val_log_AL[:, 1], 'c',
                     label='log_AL')
            plt.title('Average log likelihood between model and true dynamics '
                      'over validation data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Average log likelihood')
            plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.val_log_AL.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            name = 'Validation_standardized_average_log_likelihood'
            plt.plot(self.val_stand_log_AL[:, 0], self.val_stand_log_AL[:, 1],
                     'c', label='stand_log_AL')
            plt.title('Standardized average log likelihood between model and '
                      'true dynamics over validation data')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Standardized average log likelihood')
            plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
            file = pd.DataFrame(self.val_stand_log_AL.numpy())
            file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                        header=False)

            # Plot model evaluation scatter plot over the val data
            plot_model_evaluation(Evaluation_grid=X_val_grid,
                                  Grid_controls=U_val_grid,
                                  Predicted_grid=predicted_grid,
                                  True_predicted_grid=Y_val_grid,
                                  folder=self.validation_folder,
                                  ground_truth_approx=self.ground_truth_approx,
                                  title='Val_model_evaluation', verbose=False)
            # self.save_grid_variables(X_val, U_val, Y_val,
            #                          results_folder=self.validation_folder)
            filename = 'Predicted_Yval' + str(self.step) + '.csv'
            file = pd.DataFrame(predicted_grid.cpu().numpy())
            file.to_csv(os.path.join(self.validation_folder, filename),
                        header=False)

        # Rollouts over the val data: random subsets of val data, number
        # proportional to length of val data, or full validation scenario for
        # each rollout
        if self.step == 1:
            self.create_val_rollouts(X_val, U_val, Y_val, cut_idx)
            save_rollout_variables(self, self.val_folder,
                                   len(self.val_rollout_list),
                                   self.val_rollout_list, step=self.step - 1,
                                   results=False,
                                   ground_truth_approx=self.ground_truth_approx,
                                   title='Val_rollouts',
                                   plots=self.monitor_experiment)  # TODO
        self.evaluate_val_rollouts()
