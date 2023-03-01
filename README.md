# Learning dynamics from partial observations: ML + observer design

## Installation

- create a directory for this repo, further named dir
- create dir/Figures/Logs
- clone the repo in dir/repo
- unzip the data files and move them to dir/Data/...
- create a **Python 3.9.16** virtual environment for this repo in 
  dir/venv, source it (for example with `python3.9 -m venv venv`)
- install all requirements with pip: `pip install -r repo/official_requirements.txt`
- if any problems occur during the installation of required packages, see
  requirements.txt for possible fixes
- install interpolation repo: `git clone https://github.
  com/aliutkus/torchinterp1d` into dir, `cd torchinterp1d`, `pip install -e .
  `, then `git checkout a29661bae59e8b3db6ec272eeaddff256026f383` to go back 
  to the commit corresponding to the versions in the requirements
- `cd ../repo`, run `python mass_spring_mass_testcase/MSM_observer_GP.py 1` (1 =
  process number, used for logging)

## Content

1. Code for project HGO+GP: paper "Joint state and dynamics estimation with 
   high-gain observers and Gaussian process models"
Combined state and dynamics estimation
with high-gain observers and Gaussian process models. Some unknown dynamical 
system is transformed into the canonical 
observable form with an unknown nonlinear part, which is learned with a GP 
from noisy, partial measurements (only the first dimension of the state is 
measured) using an adaptive high-gain observer. After each cycle, the GP 
estimation is used to update the observer.
Code in `GP_models` and all scripts containing `observer_GP` in their name.


2. Code for project recognition_NODEs: paper "Recognition models to learn 
   dynamics from partial observations with Neural ODEs"
Approximation of the ODE generating 
some data with a NN, learning it jointly with a recognition model that maps 
the partial observations to the initial latent state. 
Code in NN_for_ODEs and all scripts containing `NN` or `difftraj` in their name.

Contains tests with simulation and real data. 
The files Data/My_wholefield3_Fluent_elbow_testcase.zip and 
Data/My_wholefield3_Fluent_elbow_testcase_outsideHeatTransfer8.zip contain 
the data for the HGO+GP Fluent test case; property of Ansys Inc.
The file `Data/Wandercraft_id.zip` contains 
the data for the NODE Wandercraft test case; property of Wandercraft.

The folder `simulation` contains the code to simulate different systems, 
including system definitions and observers. The other folders contain 
the scripts for the use case indicated in the name, such as the folder 
`benchmark_recognition_models` for the benchmark of different recognition 
models with NODEs. In this folder, run `earthquake_difftraj.py` for a fixed 
dynamics model of the Earthquake system learning only the recognition model, 
`earthquake_difftraj_paramid.py` for a parametric dynamics model and 
learning the recognition model while jointly optimizing the parameters, and 
`earthquake_difftraj_fullNODE.py` for learning a full NODE model jointly with 
the recognition model. Each script is roughly described at the beginning of 
the code, and each name should be rather explicit when compared to the 
experiments in the papers or in the manuscript.


## Usage

To run any script in the repo, for example `MSM_observer_GP`, learning a 
Gaussian process model of the mass-spring-mass test case, run `python 
mass_spring_mass_testcase/MSM_observer_GP.py 1` where 1 is the process 
number used for logging (when running different scripts in parallel, use 
different numbers). Change the scripts in the folder or the parameters in the 
script to explore different options. 

1. To reproduce the results of the paper: "Joint state and dynamics estimation with 
   high-gain observers and Gaussian process models"

- Mass-spring-mass test case: run `python 
mass_spring_mass_testcase/MSM_observer_GP.py 1`. The main methods to select are 
  in the name of the system: `GP_Michelangelo_highgain_observer_noisy_inputs` 
  for the method in "Model Identification and Adaptive State Observation for 
  a Class of Nonlinear Systems" on which our paper is built, 
  `GP_justvelocity_highgain_observer_noisy_inputs` for the method described 
  in our paper, and `GP_justvelocity_adaptive_highgain_observer_noisy_inputs` for the 
  method described in our paper but with adaptive gain.
- Duffing test case: run `python duffing_testcase/Duffing_observer_GP.py 1` 
  for the HGO and GP model, `python 
  duffing_testcase/quasilinear_observer_leastsquares.py 1` for the HGO 
  combined with a parametric model of the oscillator whose parameters are 
  obtained by least squares estimation, as in "Model Identification and Adaptive State Observation for 
  a Class of Nonlinear Systems".
- Fluent test case: run `fluent_elbow_testcase/fluent_elbow_no_observer_GP.
  py` to learn the prior GP model from the provided data, and 
  `fluent_elbow_testcase/fluent_elbow_observer_GP.py` to refine it from 
  measurements at a single point of the modified dataset. 

Change the parameters (names are usually explicit) to explore different 
options. 


2. To reproduce the results of the paper: "Recognition models to learn 
   dynamics from partial observations with Neural ODEs"

- Benchmark of recognition models: run `python 
  benchmark_recognition_models/earthquake_difftraj_fullNODE.py 1 
  KKL_u0T_back` for the earthquake model with joint optimization of the 
  dynamical parameters (Fig. 2, left). Similarly for the 
  FitzHugh-Nagumo model (Fig. 2, middle), run `python 
  benchmark_recognition_models/FitzHugh_Nagumo_ODE_difftraj_fullNODE.py 1 
  KKL_u0T_back`, and for the Van der Pol model (Fig. 2, right), run `python 
  benchmark_recognition_models/vanderpol_difftraj_fullNODE.py 1 
  KKLu_back`. Options for the recognition model (second argument of the 
  python command, the first being the process number) are: `KKL_u0T_back` for 
  backward KKL (`KKL_u0T` for forward), `KKLu_back` for 
  backward KKLu (`KKLu` for forward), `y0T_u0T` for direct, `y0_u0` for 
  direct with t_c = 0, `y0T_u0T_RNN_outNN_back` for backward RNN+ 
  (`y0T_u0T_RNN_outNN` for forward). Modify the other parameters (mostly 
  `init_state_obs_T` and `true_meas_noise_var`) to reproduce the ablation 
  studies.
- Harmonic oscillator: run `python 
  harmonic_oscillator_testcase/HO_back_physical_coords_NN_difftraj.py 1 
  KKL_u0T_back` for no structure (Fig. 5 a), 
  `HO_back_physical_coords_NN_hamiltonian.py` for Hamiltonian (Fig. 5 (b)), 
  `HO_back_physical_coords_NN_hamiltonian_x1dotx2.py` for a particular 
  Hamiltonian (Fig. 5 (c)), `HO_back_physical_coords_NN_paramid_linobs.py` for 
  a parametric model (Fig. 5 (d)), and `HO_back_physical_coords_NN_only_recog.
  py` for the extended state-space model (Fig. 5 (e)). The main options for the 
  recognition method (second argument) are again: `KKL_u0T_back`, `y0T_u0T_RNN_outNN_back`, 
  `y0T_u0T`, `y0_u0`.
- Robotic exoskeleton: run `python wandercraft_id/wandercraft_id_difftraj.py 
  1 KKL_u0T_back` for no structure (Fig. 7 (b)), 
  `wandercraft_id_difftraj_x1dotx2.py` for x1dot = x2, x3dot = x4 (Fig. 7 (b))
  , `wandercraft_id_difftraj_x1dotx2_residuals.py` for the residuals of the linear prior on 
  top of this constraint (Fig. 7 (c)). The options for the recognition 
  method (second argument) are: `KKL_u0T_back`, `y0T_u0T_RNN_outNN_back`, 
  `y0T_u0T`, `y0_u0`, `KKLu_back`.

The code runs in a few minutes on a regular laptop for the first two cases, 
but will need about a day for the robotics dataset.


## If you use this repo, please cite:
For HGO+GP:
```
@article{HGO_GP,
author = {Buisson-Fenet, Mona and Morgenthaler, Valery and Trimpe, Sebastian and {Di Meglio}, Florent},
journal = {IEEE Control Systems Letters},
number = {5},
pages = {1627--1632},
title = {{Joint state and dynamics estimation with high-gain observers and Gaussian process models}},
volume = {5},
year = {2021}
}
```

For NODEs:
```
@article{recognition_NODEs,
	author = {Buisson-Fenet, Mona and Morgenthaler, Valery and Trimpe, Sebastian and {Di Meglio}, Florent},
	journal = {Transactions on Machine Learning Research},
	title = {{Recognition Models to Learn Dynamics from Partial Observations with Neural ODEs}},
	url = {https://openreview.net/forum?id=LTAdaRM29K},
	year = {2023}
}
```
 
