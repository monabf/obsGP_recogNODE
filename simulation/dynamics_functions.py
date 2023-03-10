import numpy as np
import torch
import copy
from functorch import vmap, jacrev

from utils.utils import reshape_pt1, reshape_pt1_tonormal, reshape_dim1_tonormal

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Possible dynamics functions as classes, calling the object of each class
# returns dx/dt


# Dynamics of the continuous time Duffing oscillator, with control law u(t)
class Duffing:

    def __init__(self, device, kwargs):
        self.device = device
        self.alpha = kwargs.get('alpha')
        self.beta = kwargs.get('beta')
        self.delta = kwargs.get('delta')
        self.A = torch.tensor([[0., 1.], [-self.alpha, -self.delta]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.duffing_dynamics(t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control)

    def duffing_dynamics(self, t, x, u, t0, init_control, process_noise_var,
                         kwargs, impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        F = torch.tensor([
            torch.zeros_like(x[:, 0]),
            - self.beta * torch.pow(x[:, 0], 3) + reshape_pt1_tonormal(u)],
            device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Dynamics of the continuous time Van der Pol oscillator, with control law u(t)
# See http://www.tangentex.com/VanDerPol.htm
class VanDerPol:

    def __init__(self, device, kwargs):
        self.device = device
        self.mu = kwargs.get('mu')
        self.A = torch.tensor([[0., 1.], [-1., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.VanderPol_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def VanderPol_dynamics(self, t, x, u, t0, init_control,
                           process_noise_var, kwargs,
                           impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += self.mu * (1 - x[:, 0] ** 2) * x[:, 1] + u[:, 0]
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def VanderPol_dynamics_xu(self, x, u):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] += x[..., 1]
        xdot[..., 1] += self.mu * (1 - x[..., 0] ** 2) * x[..., 1] \
                        - x[..., 0] + u[..., 0]
        return xdot

# Dynamics of the continuous time Lorenz oscillator, with control law u(t)
# See https://en.wikipedia.org/wiki/Lorenz_system
# https://arxiv.org/pdf/2201.05136.pdf
# https://github.com/josephbakarji/deep-delay-autoencoder/blob/main/examples/lorenz.py
class Lorenz:

    def __init__(self, device, kwargs):
        self.device = device
        self.sigma = kwargs.get('sigma')
        self.rho = kwargs.get('rho')
        self.beta = kwargs.get('beta')
        self.A = torch.tensor([[-self.sigma, self.sigma, 0.],
                               [self.rho, -1., 0.],
                               [0., 0., -self.beta]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.Lorenz_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def Lorenz_dynamics(self, t, x, u, t0, init_control,
                           process_noise_var, kwargs,
                           impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[..., 1] -= x[..., 0] * x[..., 2]
        xdot[..., 2] += x[..., 0] * x[..., 1]
        xdot += u
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def Lorenz_dynamics_xu(self, x, u):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] += self.sigma * (x[..., 1] - x[..., 0])
        xdot[..., 1] += x[..., 0] * (self.rho - x[..., 2]) - x[..., 1]
        xdot[..., 2] += x[..., 0] * x[..., 1] - self.beta * x[..., 2]
        xdot += u
        return xdot


# Dynamics of a simple inverted pendulum, with control law u(t), continuous time
# http://www.matthewpeterkelly.com/tutorials/simplePendulum/index.html
class Pendulum:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.g = kwargs.get('g')
        self.l = kwargs.get('l')
        self.A = torch.tensor([[0., 1.], [0., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.pendulum_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def pendulum_dynamics(self, t, x, u, t0, init_control, process_noise_var,
                          kwargs, impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        theta_before = x[:, 0]
        thetadot_before = x[:, 1]
        F = torch.tensor([[np.zeros_like(x[:, 0]),
                           - self.g / self.l * np.sin(theta_before) -
                           self.k / self.m * thetadot_before]], device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F + u
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Dynamics of a harmonic oscillator, with control law u(t), continuous time,
# pulsation/angular frequency w (2pi/period)
# https://en.wikipedia.org/wiki/Harmonic_oscillator
class Harmonic_oscillator:

    def __init__(self, device, kwargs):
        self.device = device
        self.w = kwargs.get('pulse')
        self.A = torch.tensor([[0., 1.], [- self.w ** 2, 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.harmonic_oscillator_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def harmonic_oscillator_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def harmonic_oscillator_extended_dynamics(self, t, x, u, t0,
                                              init_control,
                                              process_noise_var, kwargs,
                                              impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.zeros_like(x)
        xdot[:, 0] = x[:, 1]
        xdot[:, 1] = - x[:, 2] * x[:, 0] + reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Classic form of the mass-spring-mass system
class MSM:

    def __init__(self, device, kwargs):
        self.device = device
        self.nx = 4
        self.n = 4
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.k1 = kwargs.get('k1')
        self.k2 = kwargs.get('k2')
        self.Ax = torch.tensor(
            [[0., 1, 0, 0], [0., 0, 0, 0], [0, 0, 0, 1.], [0, 0, 0, 0.]],
            device=self.device)
        self.Bx = torch.tensor([[0.], [0], [0], [1]], device=self.device)
        self.A = torch.tensor(
            [[0., 1, 0, 0], [0., 0, 1, 0], [0, 0, 0, 1.], [0, 0, 0, 0.]],
            device=self.device)
        self.F1 = reshape_pt1(torch.zeros((self.n - 1,), device=self.device))

    def __call__(self, t, z, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.mass_spring_mass_dynamics_z(t, z, u, t0, init_control,
                                                process_noise_var, kwargs,
                                                impose_init_control)

    def mass_spring_mass_dynamics_x(self, t, x, u, t0, init_control,
                                    process_noise_var, kwargs,
                                    impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        F = torch.tensor([
            torch.zeros_like(x[:, 0]),
            self.k1 / self.m1 * (x[:, 2] - x[:, 0]) +
            self.k2 / self.m1 * (x[:, 2] - x[:, 0]) ** 3,
            torch.zeros_like(x[:, 2]),
            -self.k1 / self.m2 * (x[:, 2] - x[:, 0]) -
            self.k2 / self.m2 * (x[:, 2] - x[:, 0]) ** 3],
            device=device)
        xdot = (torch.matmul(self.Ax.double(), x.double().t()) +
                torch.matmul(self.Bx.double(), u.double())).t()
        xdot += reshape_pt1(F)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), xdot.shape,
                                 device=device)
        return xdot

    # Canonical form of the mass-spring-mass system using x1 as flat output
    def mass_spring_mass_dynamics_z(self, t, z, u, t0, init_control,
                                    process_noise_var, kwargs,
                                    impose_init_control=False):
        device = z.device
        z = reshape_pt1(z)
        z3 = reshape_pt1(z[:, 2])
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        v = reshape_pt1_tonormal(self.mass_spring_mass_v(z, kwargs))
        vdot = reshape_pt1_tonormal(self.mass_spring_mass_vdot(z, kwargs))
        F2 = torch.tensor([[
            self.k1 / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3) +
            (3 * self.k2) / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3)
            * v ** 2 + (6 * self.k2) / self.m1 * v * vdot ** 2]], device=device)
        F = reshape_pt1(torch.cat((self.F1, F2), dim=1))
        zdot = torch.matmul(self.A, z.double().t()).t() + F
        if process_noise_var != 0:
            zdot += torch.normal(0, np.sqrt(process_noise_var), size=zdot.shape,
                                 device=device)
        return zdot

    # Canonical form of the mass-spring-mass system using x1 as flat output,
    # only last dimension
    def mass_spring_mass_dynamics_z_justvelocity(self, t, z, u, t0,
                                                 init_control,
                                                 process_noise_var, kwargs,
                                                 impose_init_control=False):
        device = z.device
        z = reshape_pt1(z)
        z3 = reshape_pt1(z[:, 2])
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        v = reshape_pt1_tonormal(self.mass_spring_mass_v(z, kwargs))
        vdot = reshape_pt1_tonormal(self.mass_spring_mass_vdot(z, kwargs))
        zdot = torch.tensor([[
            self.k1 / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3) +
            (3 * self.k2) / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3)
            * v ** 2 + (6 * self.k2) / self.m1 * v * vdot ** 2]], device=device)
        if process_noise_var != 0:
            zdot += torch.normal(0, np.sqrt(process_noise_var), size=zdot.shape,
                                 device=device)
        return zdot

    # Utility function for the mass-spring-mass system
    # Solution obtained with http://eqworld.ipmnet.ru/en/solutions/ae/ae0103.pdf
    def mass_spring_mass_v(self, z, kwargs):
        z = reshape_pt1(z)
        x1d2 = reshape_pt1(z[:, 2])
        p = self.k1 / self.k2
        q = - self.m1 / self.k2 * x1d2
        D = np.power(p / 3, 3) + np.power(q / 2, 2)
        A = np.cbrt(-q / 2 + np.sqrt(D))  # np.power not with negative floats!
        B = np.cbrt(-q / 2 - np.sqrt(D))
        v = reshape_pt1(A + B)
        return v

    # Utility function for the mass-spring-mass system
    def mass_spring_mass_vdot(self, z, kwargs):
        z = reshape_pt1(z)
        x1d3 = reshape_pt1(z[:, 3])
        A = self.k1 / self.m1 + \
            3 * self.k2 / self.m1 * self.mass_spring_mass_v(z, kwargs) ** 2
        vdot = reshape_pt1(x1d3 / A)
        return vdot

    # Flat transform (from x to z) for mass-spring-mass system
    def mass_spring_mass_xtoz(self, x, kwargs):
        x = reshape_pt1(x)
        z = x.clone()
        z[:, 2] = self.k1 / self.m1 * (x[:, 2] - x[:, 0]) + \
                  self.k2 / self.m1 * (x[:, 2] - x[:, 0]) ** 3
        z[:, 3] = self.k1 / self.m1 * (x[:, 3] - x[:, 1]) + \
                  3 * self.k2 / self.m1 * (x[:, 3] - x[:, 1]) * (
                          x[:, 2] - x[:, 0]) ** 2
        return reshape_pt1(z)

    # Inverse transform (from z to x) for mass-spring-mass system
    def mass_spring_mass_ztox(self, z, kwargs):
        z = reshape_pt1(z)
        x = z.clone()
        x[:, 2] = reshape_pt1_tonormal(
            self.mass_spring_mass_v(z, kwargs)) + z[:, 0]
        x[:, 3] = reshape_pt1_tonormal(
            self.mass_spring_mass_vdot(z, kwargs)) + z[:, 1]
        return reshape_pt1(x)


# Standard form of the continuous time reverse Duffing oscillator, with u(t)
class Reverse_Duffing:

    def __init__(self, device, kwargs):
        self.device = device

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.reverse_duffing_dynamics_z(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def reverse_duffing_dynamics_x(self, t, x, u, t0, init_control,
                                   process_noise_var, kwargs,
                                   impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = torch.empty_like(x)
        xdot[:, 0] = torch.pow(x[:, 1], 3)
        xdot[:, 1] = - x[:, 0] + reshape_pt1(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    # Canonical form of the continuous time reverse Duffing oscillator, with u(t)
    def reverse_duffing_dynamics_z(self, t, x, u, t0, init_control,
                                   process_noise_var, kwargs,
                                   impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1_tonormal(u(t, kwargs, t0, init_control,
                                   impose_init_control))
        xdot = torch.tensor([x[:, 1], 3 * torch.pow(
            torch.abs(x[:, 1]), 2. / 3) * (u - x[:, 0])], device=device)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    # Flat transform (from x to z) for reverse Duffing
    def reverse_duffing_xtoz(self, x):
        x = reshape_pt1(x)
        z = x.clone()
        z[:, 1] = torch.pow(x[:, 1], 3)
        return reshape_pt1(z)

    # Inverse transform (from z to x) for reverse Duffing
    def reverse_duffing_ztox(self, z):
        z = reshape_pt1(z)
        x = z.clone()
        x[:, 1] = torch.sign(z[:, 1]) * torch.pow(torch.abs(z[:, 1]), 1. / 3)
        return reshape_pt1(x)

    # True prior mean of only velocity (for backPhi with HGO and true prior)
    def reverse_duffing_dynamics_z_justvelocity_true(self, x, u, prior_kwargs):
        x = reshape_pt1(x)
        u = reshape_pt1_tonormal(u)
        vdot = 3 * torch.pow(torch.abs(x[:, 1]), 2. / 3) * (u - x[:, 0])
        return reshape_pt1(vdot)


# Dynamics of a building during an earthquake, with control law u(t),
# continuous time
# https://odr.chalmers.se/bitstream/20.500.12380/256887/1/256887.pdf
class Earthquake_building:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.A = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 0.],
                               [0., 0., 0., 1.], [0., 0., 0., 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += self.k / self.m * (x[:, 2] - 2 * x[:, 0]) - \
                      reshape_pt1_tonormal(u)
        xdot[:, 3] += self.k / self.m * (x[:, 0] - x[:, 2]) - \
                      reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        u[..., 0]
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) - u[..., 0]
        return xdot

# Same as Earthquake_building except the input created by the earthquake is
# considered as a time-dependent nonlinear perturbation instead of a control
# input, so extended state to take time = x5 into account in the state with
# tdot = 1. Constant control input actually contains parameters of
# perturbation to allow for different earthquakes on same model easily
class Earthquake_building_timedep:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.A = torch.tensor([[0., 1., 0., 0., 0.], [0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0.], [0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        gamma = u[:, 0]
        omega = u[:, 1]
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += self.k / self.m * (x[:, 2] - 2 * x[:, 0]) - \
                      gamma * torch.cos(omega * x[:, 4])
        xdot[:, 3] += self.k / self.m * (x[:, 0] - x[:, 2]) - \
                      gamma * torch.cos(omega * x[:, 4])
        xdot[:, 4] = torch.ones_like(xdot[:, 4])
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        gamma = u[..., 0]
        omega = u[..., 1]
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        gamma * torch.cos(omega * x[..., 4])
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) - \
                        gamma * torch.cos(omega * x[..., 4])
        xdot[..., 4] = torch.ones_like(xdot[..., 4])
        return xdot

# Same as Earthquake_building except the input created by the earthquake is
# generated by a harmonic oscillator. Constant control input actually
# contains parameters of perturbation to allow for different earthquakes on
# same model easily
class Earthquake_building_extended:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.t0 = kwargs.get('t0')
        # cheat to overwrite x5(t0) and x6(t0) with constant control
        self.A = torch.tensor([
            [0., 1., 0., 0., 0., 0.],
            [-2 * self.k / self.m, 0., self.k / self.m, 0., -1., 0.],
            [0., 0., 0., 1., 0., 0.],
            [self.k / self.m, 0., - self.k / self.m, 0., -1., 0.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        omega = u[:, 1]
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 5] = - omega ** 2 * x[:, 4]
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        omega = u[..., 1]
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        x[..., 4]
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) -\
                        x[..., 4]
        xdot[..., 4] = x[..., 5]
        xdot[..., 5] = - omega ** 2 * x[..., 4]
        return xdot


# ODE version of FitzHugh-Nagumo model of the evolution of the
# electromagnetic potential through a membrane subject to a stimulus
# https://en.wikipedia.org/wiki/FitzHugh???Nagumo_model
# Optimal control for estimation in partially observed elliptic and
# hypoelliptic linear stochastic differential equations, Q. Clairon, A. Samson
class FitzHugh_Nagumo_ODE:

    def __init__(self, device, kwargs):
        self.device = device
        self.eps = kwargs.get('eps')
        self.gamma = kwargs.get('gamma')
        self.beta = kwargs.get('beta')
        self.A = torch.tensor([[1 / self.eps, - 1 / self.eps],
                               [self.gamma, -1.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.FitzHugh_Nagumo_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def FitzHugh_Nagumo_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 0] += - 1 / self.eps * torch.pow(x[:, 0], 3) + u
        xdot[:, 1] += self.beta
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def FitzHugh_Nagumo_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] += 1 / self.eps * (x[..., 0] - x[..., 1] -
                                        torch.pow(x[..., 0], 3)) + u[..., 0]
        xdot[..., 1] += self.gamma * x[..., 0] - x[..., 1] + self.beta * \
                        torch.ones_like(x[..., 1])
        return xdot


class QuanserQubeServo2:
    """ See https://www.quanser.com/products/qube-servo-2/ QUBE SERVO 2 and
    for a detailed reference for this system.
    Documentation on the simulator:
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/gym_brt/quanser/qube_simulator.py
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/tests/notebooks/Compare%20Qube%20Hardware%20to%20ODEint.ipynb

    State: (theta, alpha, theta_dot, alpha_dot)
    Measurement: (theta, alpha)
    """

    def __init__(self):
        super().__init__()
        self.dim_x = 4
        self.dim_y = 2

        # Motor
        # self.Rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        # self.km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        self.mr = 0.095  # Mass (kg)
        self.Lr = 0.085  # Total length (m)
        self.Jr = self.mr * self.Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        # self.Dr = 0.00027  # Equivalent viscous damping coefficient (N-m-s/rad)

        # Pendulum Link
        self.mp = 0.024  # Mass (kg)
        self.Lp = 0.129  # Total length (m)
        self.Jp = self.mp * self.Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        # self.Dp = 0.00005  # Equivalent viscous damping coefficient (N-m-s/rad)

        # After identification on hardware data:
        self.Rm = 14
        self.km = 0.01
        self.Dr = 0.0005
        self.Dp = -3e-5

        self.gravity = 9.81  # Gravity constant

    def qube_dynamics_xu(self, x, action=0):
        theta = x[..., 0]
        alpha = x[..., 1]
        theta_dot = x[..., 2]
        alpha_dot = x[..., 3]

        Vm = action
        tau = -(self.km * (Vm - self.km * theta_dot)) / self.Rm

        xdot = torch.zeros_like(x)
        xdot[..., 0] = theta_dot
        xdot[..., 1] = alpha_dot
        xdot[..., 2] = (
            -self.Lp
            * self.Lr
            * self.mp
            * (
                -8.0 * self.Dp * alpha_dot
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * torch.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * torch.sin(alpha)
            )
            * torch.cos(alpha)
            + (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Dr * theta_dot
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * torch.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * torch.sin(alpha)
                - 4.0 * tau
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * torch.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
        )

        xdot[..., 3] = (
            2.0
            * self.Lp
            * self.Lr
            * self.mp
            * (
                4.0 * self.Dr * theta_dot
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * torch.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * torch.sin(alpha)
                - 4.0 * tau
            )
            * torch.cos(alpha)
            - 0.5
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
            * (
                -8.0 * self.Dp * alpha_dot
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * torch.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * torch.sin(alpha)
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * torch.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
        )

        return xdot

    # For flexibility and coherence: use remap function after every simulation
    # But be prepared to change its behavior!
    def remap_angles(self, traj):
        # Map theta to [-pi,pi] and alpha to [0, 2pi]
        traj[..., 0] = ((traj[..., 0] + np.pi) % (2 * np.pi)) - np.pi
        traj[..., 1] = traj[..., 1] % (2 * np.pi)
        return traj

    # For adapting hardware data to the conventions of the simulation model
    def remap_hardware_angles(self, traj, add_pi_alpha=False):
        # Reorder as (theta, alpha, thetadot, alphadot)
        # Convention for alpha: 0 is upwards (depends on dataset!)
        # Remap as simulation data
        traj_copy = copy.deepcopy(traj)
        traj[..., 0], traj[..., 1] = traj_copy[..., 1], traj_copy[..., 0]
        traj[..., 2], traj[..., 3] = traj_copy[..., 3], traj_copy[..., 2]
        if add_pi_alpha:
            traj[..., 1] += np.pi
        return self.remap_angles(traj)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        return self.qube_dynamics_xu(x, u)

    def call_deriv(self, t, x, u, t0, init_control, process_noise_var,
                   kwargs, impose_init_control=False):
        return self.predict_deriv(x, self.qube_dynamics_xu)

    # Jacobian of function that predicts xt+dt (or other) from xt and the
    # function u. Useful for EKF!
    def predict_deriv(self, x, f):
        # Compute Jacobian of f with respect to input x
        # TODO: more efficient computation dNN/dx(x)! Symbolic? JAX?
        # dfdh = torch.autograd.functional.jacobian(
        #     f, x, create_graph=False, strict=False, vectorize=False)
        dfdh = vmap(jacrev(f))(x)
        dfdx = torch.squeeze(dfdh)
        return dfdx
