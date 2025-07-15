import torch

import sys
sys.path.append('C:\project\iLQR\mpc.pytorch.mine\mpc')
from mpc_explicit import MPC
from mpc_explicit import GradMethods, QuadCost
from dynamics import NNDynamics
import util as eutil
from env_dx import pendulum, cartpole

import numpy as np
import numpy.random as npr

import os
import sys
import shutil
import time

import pickle as pkl

from setproctitle import setproctitle

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector


class IL_Env:
    def __init__(self, env, lqr_iter=100, mpc_T=35, slew_rate_penalty=None):
        self.env = env

        if self.env == 'pendulum':
            self.true_dx = pendulum.PendulumDx()
        elif self.env == 'cartpole':
            self.true_dx = cartpole.CartpoleDx()
        elif self.env == 'pendulum-complex':
            params = torch.tensor((10., 1., 1., 1.0, 0.1))
            self.true_dx = pendulum.PendulumDx(params, simple=False)
        else:
            assert False

        self.lqr_iter = lqr_iter
        self.mpc_T = mpc_T
        self.slew_rate_penalty = slew_rate_penalty

        self.grad_method = GradMethods.ANALYTIC

        self.train_data = None
        self.val_data = None
        self.test_data = None



    def sample_xinit(self, n_batch=1):
        def uniform(shape, low, high):
            r = high-low
            return torch.rand(shape)*r+low

        if self.env in ['pendulum', 'pendulum-complex']:
            th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
            thdot = uniform(n_batch, -1., 1.)
            xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)
        elif self.env == 'cartpole':
            # qpos = uniform((self.n_batch, 2), -0.1, 0.1)
            # qvel = uniform((self.n_batch, 2), -0.1, 0.1)
            # xinit = torch.cat((qpos, qvel), dim=1)
            x = uniform(n_batch, -0.5, 0.5)*0
            dx = uniform(n_batch, -0.5, 0.5)*0
            th = uniform(n_batch, -np.pi, np.pi)*0+torch.ones(n_batch)*3.1415926/1.05
            dth = uniform(n_batch, -1., 1.)*0
            xinit = torch.stack((x, dx, torch.cos(th), torch.sin(th), dth), dim=1)
        else:
            import ipdb; ipdb.set_trace()

        return xinit

    def populate_data(self, n_train, n_val, n_test, seed=0):
        torch.manual_seed(seed)

        n_data = n_train+n_val+n_test
        xinit = self.sample_xinit(n_batch=n_data)

        true_q, true_p = self.true_dx.get_true_obj()

        true_x_mpc, true_u_mpc = self.mpc(self.true_dx, xinit, true_q, true_p)
        tau = torch.cat((true_x_mpc, true_u_mpc), dim=2).transpose(0,1)

        self.train_data = tau[:n_train]
        self.val_data = tau[n_train:n_train+n_val]
        self.test_data = tau[-n_test:]

    def populate_data2(self, n_train, n_val, n_test, seed=0):
        T = self.mpc_T
        torch.manual_seed(seed)

        n_total = n_train + n_val + n_test
        dx = self.true_dx
        q, p = dx.get_true_obj()

        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
        p_repeat = p.unsqueeze(0).repeat(T, 1, 1)

        x_init_all = self.sample_xinit(n_batch=n_total)
        all_tau = []

        for i in range(n_total):
            x = x_init_all[i].unsqueeze(0)  # (1, n_state)
            u_init = None
            x_traj = [x.squeeze(0)]
            u_traj = []

            for t in range(T):
                Q_i = Q[:, :1].clone()  # (T, 1, n_state, n_state)
                p_i = p_repeat[:, :1].clone()  # (T, 1, n_state)

                nominal_states, nominal_actions, _ = MPC(
                    dx.n_state, dx.n_ctrl, T,
                    u_init=u_init,
                    u_lower=dx.lower, u_upper=dx.upper,
                    lqr_iter=self.lqr_iter,
                    verbose=0,
                    exit_unconverged=False,
                    detach_unconverged=False,
                    linesearch_decay=dx.linesearch_decay,
                    max_linesearch_iter=dx.max_linesearch_iter,
                    grad_method=self.grad_method,
                    eps=dx.mpc_eps,
                )(x, QuadCost(Q_i, p_i), dx)

                next_action = nominal_actions[0]
                u_traj.append(next_action.squeeze(0))
                x = dx(x, next_action)
                x_traj.append(x.squeeze(0))

                u_init = torch.cat((nominal_actions[1:], torch.zeros(1, 1, dx.n_ctrl)), dim=0)
                u_init[-2] = u_init[-3]

            x_traj = torch.stack(x_traj, dim=0)  # (T+1, n_state)
            u_traj = torch.stack(u_traj, dim=0)  # (T, n_ctrl)
            tau = torch.cat((x_traj[:-1], u_traj), dim=1)  # (T, state+ctrl)
            all_tau.append(tau)

        all_tau = torch.stack(all_tau, dim=0)  # (n_total, T, state+ctrl)

        self.train_data = all_tau[:n_train]
        self.val_data = all_tau[n_train:n_train + n_val]
        self.test_data = all_tau[n_train + n_val:]

    def mpc(self, dx, xinit, q, p, u_init=None, eps_override=None,
            lqr_iter_override=None):
        n_batch = xinit.shape[0]

        n_sc = self.true_dx.n_state+self.true_dx.n_ctrl

        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            self.mpc_T, n_batch, 1, 1
        )
        p = p.unsqueeze(0).repeat(self.mpc_T, n_batch, 1)

        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter

        x_mpc, u_mpc, objs_mpc = MPC(
            self.true_dx.n_state, self.true_dx.n_ctrl, self.mpc_T,
            u_lower=self.true_dx.lower, u_upper=self.true_dx.upper, u_init=u_init,
            lqr_iter=lqr_iter,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=True,
            linesearch_decay=self.true_dx.linesearch_decay,
            max_linesearch_iter=self.true_dx.max_linesearch_iter,
            grad_method=self.grad_method,
            eps=eps,
            # slew_rate_penalty=self.slew_rate_penalty,
            # prev_ctrl=prev_ctrl,
        )(xinit, QuadCost(Q, p), dx)
        return x_mpc, u_mpc

