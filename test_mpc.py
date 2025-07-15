import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset, DataLoader
import gym

import sys
sys.path.append('../')
from mpc_explicit import MPC
from mpc_explicit import GradMethods
# #from mpc.dynamics import NNDynamics
# #import mpc.util as eutil
from env_dx import pendulum, cartpole
from definitions import QuadCost

import numpy as np
import numpy.random as npr
import argparse
import os
import sys
import time

import pickle as pkl

env=gym.make('Pendulum-v1')
obs = env.reset()
# env.state = np.array([np.pi, 0.])  # 设置自定义状态
dx = pendulum.PendulumDx()
n_batch, T = 1,20
xinit = torch.zeros(n_batch, dx.n_state)
# xinit[:,0] = np.cos(np.pi)
# xinit[:,1] = np.sin(np.pi)

xinit[:, 0] = torch.tensor(obs[0], dtype=torch.float32)  # cos(theta)
xinit[:, 1] = torch.tensor(obs[1], dtype=torch.float32)  # sin(theta)
xinit[:, 2] = torch.tensor(obs[2], dtype=torch.float32)
x = xinit
u_init = None

q,p=dx.get_true_obj()
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
    T, n_batch, 1, 1
)
p = p.unsqueeze(0).repeat(T, n_batch, 1)

mode = 'swingup'
mode = 'spin'

if mode == 'swingup':
    goal_weights = torch.Tensor((1., 1., 0.1))
    goal_state = torch.Tensor((1., 0. ,0.))
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones(dx.n_ctrl)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(T, n_batch, 1)
elif mode == 'spin':
    Q = 0.001*torch.eye(dx.n_state+dx.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = torch.tensor((0., 0., -1., 0.))
    p = p.unsqueeze(0).repeat(T, n_batch, 1)

#mpc=MPC(n_state=dx.n_state,n_ctrl=dx.n_ctrl,T=T,grad_method=GradMethods.AUTO_DIFF,eps=dx.mpc_eps,lqr_iter=500)
mpc=MPC(
        dx.n_state, dx.n_ctrl, T,
        u_init=u_init,
        u_lower=None, u_upper=None,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.ANALYTIC,
        eps=1e-2,
    )
x_mpc, u_mpc,_=mpc(xinit,QuadCost(Q,p),dx)
# print(x_mpc.shape)
# print(u_mpc.shape)
#
# x_true=torch.ones(T,n_batch,dx.n_state)
# u_true=torch.ones(T,n_batch,dx.n_ctrl)

for _ in range(200):  # Run for 200 steps or any desired number
    # Use MPC to compute the control action
    x_mpc, u_mpc, _ = mpc(xinit, QuadCost(Q, p), dx)

    # Apply the first control from the computed sequence to the environment
    u = u_mpc[0, 0].detach().numpy()
    obs, reward, done, _ = env.step(u)

    # Update xinit with the new state from the environment (convert to torch tensors)
    xinit[:, 0] = torch.tensor(obs[0], dtype=torch.float32)
    xinit[:, 1] = torch.tensor(obs[1], dtype=torch.float32)
    xinit[:, 2] = torch.tensor(obs[2], dtype=torch.float32)  # Update angular velocity
    # Render the environment
    env.render()

    if done:
        break

env.close()

# loss = (u_true-u_mpc).pow(2).mean()
# loss.backward()
# print(dx.params.grad)