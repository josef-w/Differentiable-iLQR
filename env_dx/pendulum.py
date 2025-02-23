import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

import sys
# sys.path.append('../')
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from .. import util
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)
sys.path.append('C:\project\iLQR\mpc.pytorch.mine\mpc')
import util
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class PendulumDx(nn.Module):
    def __init__(self, params=None, simple=True):
        super().__init__()
        self.simple = simple

        self.max_torque = 2.0
        self.dt = 0.05
        self.n_state = 3
        self.n_ctrl = 1

        if params is None:
            if simple:
                # gravity (g), mass (m), length (l)
                self.params = torch.tensor((10., 1., 1.), requires_grad=True)
            else:
                # gravity (g), mass (m), length (l), damping (d), gravity bias (b)
                self.params = Variable(torch.Tensor((10., 1., 1., 0., 0.)))
        else:
            self.params = params

        assert len(self.params) == 3 if simple else 5

        self.goal_state = torch.Tensor([1., 0., 0.])
        self.goal_weights = torch.Tensor([1., 1., 0.1])
        self.ctrl_penalty = 0.001
        self.lower, self.upper = -2., 2.

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 3
        assert u.shape[1] == 1
        assert u.ndimension() == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        if not hasattr(self, 'simple') or self.simple:
            g, m, l = torch.unbind(self.params)
        else:
            g, m, l, d, b = torch.unbind(self.params)

        u = torch.clamp(u, -self.max_torque, self.max_torque)[:,0]
        cos_th, sin_th, dth = torch.unbind(x, dim=1)
        th = torch.atan2(sin_th, cos_th)
        if not hasattr(self, 'simple') or self.simple:
            newdth = dth + self.dt*(-3.*g/(2.*l) * (-sin_th) + 3. * u / (m*l**2))
        else:
            sin_th_bias = torch.sin(th + b)
            newdth = dth + self.dt*(
                -3.*g/(2.*l) * (-sin_th_bias) + 3. * u / (m*l**2) - d*th)
        newth = th + newdth*self.dt
        state = torch.stack((torch.cos(newth), torch.sin(newth), newdth), dim=1)

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 3
        l = self.params[2].item()

        cos_th, sin_th, dth = torch.unbind(x)
        th = np.arctan2(sin_th, cos_th)
        x = sin_th*l
        y = cos_th*l

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        ax.plot((0,x), (0, y), color='k')
        ax.set_xlim((-l*1.2, l*1.2))
        ax.set_ylim((-l*1.2, l*1.2))
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)

    def batch_align(self,matrix_n_part_n_):
        batch_size = len(matrix_n_part_n_[0][0])
        matrix_n_part_n_1 = torch.stack(matrix_n_part_n_[0])
        matrix_n_part_n_2 = torch.stack(matrix_n_part_n_[1])
        if len(matrix_n_part_n_[2]) == 4:
            matrix_n_part_n_3 = torch.stack((torch.tensor([matrix_n_part_n_[2][0]] * batch_size).unsqueeze(-1),
                                             torch.tensor([matrix_n_part_n_[2][1]] * batch_size).unsqueeze(-1),
                                             torch.tensor([matrix_n_part_n_[2][2]] * batch_size).unsqueeze(-1),
                                             torch.tensor([matrix_n_part_n_[2][3]] * batch_size).unsqueeze(-1)))
            return torch.cat((matrix_n_part_n_1, matrix_n_part_n_2, matrix_n_part_n_3), dim=2).permute(1, 2, 0)
        elif len(matrix_n_part_n_[2]) == 3:
            if (type(matrix_n_part_n_[2][0]) == type(torch.ones(1))):
                matrix_n_part_n_3 = torch.stack(matrix_n_part_n_[2])
                return torch.cat((matrix_n_part_n_1, matrix_n_part_n_2, matrix_n_part_n_3), dim=2).permute(1, 2, 0)
            else:
                matrix_n_part_n_3 = torch.stack((torch.tensor([matrix_n_part_n_[2][0]] * batch_size).unsqueeze(-1),
                                                 torch.tensor([matrix_n_part_n_[2][1]] * batch_size).unsqueeze(-1),
                                                 torch.tensor([matrix_n_part_n_[2][2]] * batch_size).unsqueeze(-1)
                                                 ))
                return torch.cat((matrix_n_part_n_1, matrix_n_part_n_2, matrix_n_part_n_3), dim=2).permute(1, 2, 0)
        elif len(matrix_n_part_n_[2]) == 1:
            matrix_n_part_n_3 = torch.tensor([matrix_n_part_n_[2][0]] * batch_size).unsqueeze(-1).unsqueeze(0)
            return torch.cat((matrix_n_part_n_1, matrix_n_part_n_2, matrix_n_part_n_3), dim=0).permute(1, 0, 2)
        else:
            matrix_n_part_n_3 = None
    def get_matrices(self,x,u):
        cos_th = x[:, 0].unsqueeze(-1)
        sin_th = x[:, 1].unsqueeze(-1)
        dth = x[:, 2].unsqueeze(-1)
        dt = self.dt
        g, m, l = self.params.detach()
        D = self.batch_align([[sin_th * torch.sin(
            dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                                           cos_th ** 2 + sin_th ** 2),
                               -(cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.sin(
                                   dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                       sin_th, cos_th)), -dt * torch.sin(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
                               -3 * dt ** 2 * torch.sin(
                                   dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                       sin_th, cos_th)) / (l ** 2 * m)
                               ],
                              [
                                  -sin_th * torch.cos(
                                      dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                          sin_th,
                                          cos_th)) / (
                                          cos_th ** 2 + sin_th ** 2),
                                  (cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.cos(
                                      dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                          sin_th, cos_th)),
                                  dt * torch.cos(
                                      dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                          sin_th, cos_th)),
                                  3 * dt ** 2 * torch.cos(
                                      dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(
                                          sin_th,
                                          cos_th)) / (
                                          l ** 2 * m)
                              ],
                              [
                                  0,
                                  3 * dt * g / (2 * l),
                                  1,
                                  3 * dt / (l ** 2 * m)
                              ]
                              ])
        # Define matrix 2 (3-dimensional tensor: 3x4x3)
        # Define the three parts of matrix 2 (each is a 3x4 matrix)
        matrix_2_part_1 = self.batch_align([
    [
        0.00375 * sin_th**2 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l * (cos_th**2 + sin_th**2)),
        -0.00375 * sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l - 0.00375 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l,
        -0.0001875 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l,
        -2.8125e-5 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**3 * m))
    ],
    [
        0.00375 * sin_th**2 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l * (cos_th**2 + sin_th**2)),
        -0.00375 * sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l + 0.00375 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l,
        -0.0001875 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l,
        -2.8125e-5 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**3 * m))
    ],
    [
        0,
        0.075 / l,
        0,
        0
    ]
])
        matrix_2_part_2 = self.batch_align([
    [
        -0.0075 * sin_th * u * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2 * (cos_th**2 + sin_th**2)),
        0.0075 * u * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2),
        0.000375 * u * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2),
        0.0075 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2) + 5.625e-5 * u * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**4 * m**3)
    ],
    [
        -0.0075 * sin_th * u * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2 * (cos_th**2 + sin_th**2)),
        0.0075 * u * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2),
        0.000375 * u * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2),
        -0.0075 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m**2) + 5.625e-5 * u * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**4 * m**3)
    ],
    [
        0,
        0,
        0,
        -0.15 / (l**2 * m**2)
    ]
])
        matrix_2_part_3 = self.batch_align([
    [
        sin_th * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
        0.00375 * g * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l**2 - (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
        -0.05 * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
        -0.0075 * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m) + 0.015 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**3 * m)
    ],
    [
        sin_th * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
        -0.00375 * g * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / l**2 - (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
        -0.05 * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
        -0.0075 * (-0.00375 * g * sin_th / l**2 - 0.015 * u / (l**3 * m)) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m) - 0.015 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**3 * m)
    ],
    [
        0,
        -0.075 * g / l**2,
        0,
        -0.3 / (l**3 * m)
    ]
])

        # Combine the three parts into a 3x4x3 tensor
        D_grad_params = torch.stack([matrix_2_part_1, matrix_2_part_2, matrix_2_part_3], dim=3)
        # print(D_grad_params.shape)

        # Define matrix 3 (3-dimensional tensor: 3x4x3)
        matrix_3_part_1 = self.batch_align([
    [-2 * cos_th * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 - sin_th**2 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2,
     sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2) - (-2 * cos_th**2 / (cos_th**2 + sin_th**2)**2 + 1 / (cos_th**2 + sin_th**2)) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     0.05 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     0.0075 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m * (cos_th**2 + sin_th**2))],
    [2 * cos_th * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 - sin_th**2 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2,
     sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2) + (-2 * cos_th**2 / (cos_th**2 + sin_th**2)**2 + 1 / (cos_th**2 + sin_th**2)) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     0.05 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     0.0075 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m * (cos_th**2 + sin_th**2))],
    [0, 0, 0, 0]
])
        # Define matrix_part_2
        matrix_3_part_2 = self.batch_align([
    [-2 * sin_th**2 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 + sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2) + torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     2 * cos_th * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 - (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l)**2 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.05 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.0075 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m)],
    [2 * sin_th**2 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 + sin_th * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2) - torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     -2 * cos_th * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2)**2 - (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l)**2 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.05 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.0075 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m)],
    [0, 0, 0, 0]
])
        # Define matrix_part_3
        matrix_3_part_3 = self.batch_align([
    [0.05 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     -0.05 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.0025 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.000375 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m)],
    [0.05 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (cos_th**2 + sin_th**2),
     -0.05 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.0025 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)),
     -0.000375 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m)],
    [0, 0, 0, 0]
])

        # Combine the three parts into a 3x4x3 tensor
        D_grad_x = torch.stack([matrix_3_part_1, matrix_3_part_2, matrix_3_part_3], dim=3)
        # print(D_grad_x.shape)
        # Define matrix 4
        D_grad_u = self.batch_align([
    [
        0.0075 * sin_th * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m * (cos_th**2 + sin_th**2)),
        -0.0075 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m),
        -0.000375 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m),
        -5.625e-5 * torch.cos(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**4 * m**2)
    ],
    [
        0.0075 * sin_th * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m * (cos_th**2 + sin_th**2)),
        -0.0075 * (cos_th / (cos_th**2 + sin_th**2) + 0.00375 * g / l) * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m),
        -0.000375 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**2 * m),
        -5.625e-5 * torch.sin(0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l**2 * m)) / (l**4 * m**2)
    ],
    [
        0, 0, 0, 0
    ]
])
        D_grad_u = torch.stack([D_grad_u], dim=3)
        # print(D_grad_u.shape)

        # dxt/dtheta(3x3)

        x_grad_theta = self.batch_align([
            [-3 * dt ** 2 * sin_th * torch.sin(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     2 * l),
             3 * dt ** 2 * u * torch.sin(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     l ** 2 * m ** 2),
             -dt ** 2 * (-3 * g * sin_th / (2 * l ** 2) - 6 * u / (l ** 3 * m)) * torch.sin(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th))],

            [3 * dt ** 2 * sin_th * torch.cos(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     2 * l),
             -3 * dt ** 2 * u * torch.cos(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     l ** 2 * m ** 2),
             dt ** 2 * (-3 * g * sin_th / (2 * l ** 2) - 6 * u / (l ** 3 * m)) * torch.cos(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th))],

            [3 * dt * sin_th / (2 * l),
             -3 * dt * u / (l ** 2 * m ** 2),
             dt * (-3 * g * sin_th / (2 * l ** 2) - 6 * u / (l ** 3 * m))]
        ])
        # print(x_grad_theta.shape)

        # dx/dxt-1 (3x3)
        x_grad_xtm1 = self.batch_align([
            [sin_th * torch.sin(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2),
             -(cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.sin(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
             -dt * torch.sin(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th))],

            [-sin_th * torch.cos(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2),
             (cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.cos(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
             dt * torch.cos(
                 dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th))],

            [0, 3 * dt * g / (2 * l), 1]
        ])
        # print(x_grad_xtm1.shape)

        # dxt/dut-1 (3x1)
        x_grad_utm1 = self.batch_align([
            [-3 * dt ** 2 * torch.sin(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     l ** 2 * m)],
            [3 * dt ** 2 * torch.cos(
                dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)) / (
                     l ** 2 * m)],
            [3 * dt / (l ** 2 * m)]
        ])
        # print(x_grad_utm1.shape)
        return D, D_grad_params, D_grad_x, D_grad_u, x_grad_theta, x_grad_xtm1, x_grad_utm1
    def grad_input(self,X, U,K=None):
        """
        计算梯度输入，支持 batch 处理。

        参数:
        X: (20, 5, 3) - 时间步, batch 大小, 状态维度
        U: (20, 5, 1) - 时间步, batch 大小, 控制维度
        返回:
        grad_D: (19, 5, 3, 4, 3) - 时间步, batch 大小, 状态维度, 控制维度+状态维度, 参数维度
        grad_d: (19, 5, 3, 3) - 时间步, batch 大小, 状态维度, 状态维度, 参数维度
        """
        T, batch_size, n_state = X.shape
        _, _, n_ctrl = U.shape
        dx, du=n_state,n_ctrl
        dtotal=dx+du
        dtheta=3
        # 初始化输出列表 grad_D 和 grad_d
        grad_D = []
        grad_d = []

        _X, _U = X.reshape(T * batch_size, dx), U.reshape(T * batch_size, du)
        D, D_grad_params, D_grad_x, D_grad_u, x_grad_theta, x_grad_xtm1, x_grad_utm1 = self.get_matrices(_X, _U)
        D, D_grad_params, D_grad_x, D_grad_u = D.reshape(T, batch_size, dx, dtotal), D_grad_params.reshape(T, batch_size,dx, dtotal,dtheta), D_grad_x.reshape(T, batch_size, dx, dtotal, dx), D_grad_u.reshape(T, batch_size, dx, dtotal, du)
        x_grad_theta, x_grad_xtm1, x_grad_utm1 = x_grad_theta.reshape(T, batch_size, dx, dtheta), x_grad_xtm1.reshape(T,batch_size,dx,dx), x_grad_utm1.reshape(T, batch_size, dx, du)
        # 初始 gradxt 设为全零张量
        gradxt = torch.zeros(batch_size, n_state, n_state)
        X_U=torch.cat((X,U),dim=-1)
        d_grad_X=torch.einsum('tbnmk,tbm->tbnk',-D_grad_x,X_U)
        d_grad_U = torch.einsum('tbnmk,tbm->tbnk', -D_grad_u, X_U)

        for t in range(T):
            # 设置示例张量
            if K==None:
                ut_grad_xt = torch.zeros(batch_size, n_ctrl, n_state) ####K_t
                utm1_grad_xtm1 = torch.zeros(batch_size, n_ctrl, n_state) ####K_tm1
            else:
                ut_grad_xt=K[t]
                if t>0:
                    utm1_grad_xtm1=K[t-1]

            # 计算 gradxt
            if t>0:
                gradxtm1=gradxt
                gradxt = x_grad_theta[t] + torch.matmul(x_grad_xtm1[t] + torch.matmul(x_grad_utm1[t], utm1_grad_xtm1), gradxt)

            # 计算 gradDt
            if t<(T-1):
                gradDt = D_grad_params[t] + torch.matmul(D_grad_x[t] + torch.matmul(D_grad_u[t], ut_grad_xt.unsqueeze(1)), gradxt.unsqueeze(1))
                grad_D.append(gradDt)
            if t>0:
                ####in the t+1 step, trace back the t step
                xtm1_utm1_params = torch.cat((gradxtm1, torch.matmul(utm1_grad_xtm1, gradxtm1)), dim=1)
                xtm1_utm1 = torch.cat((X[t-1], U[t-1]), dim=-1)
                grad_dtm1=gradxt-torch.einsum('bnmk,bm->bnk', grad_D[t-1],xtm1_utm1)-torch.matmul(D[t-1],xtm1_utm1_params)
                grad_d.append(grad_dtm1)

        # 在循环结束后，将列表转换为 tensor
        grad_D = torch.stack(grad_D, dim=0)
        grad_d = torch.stack(grad_d, dim=0)

        return grad_D, grad_d,D_grad_x[:T-1], D_grad_u[:T-1],D[:T-1],d_grad_X[:T-1],d_grad_U[:T-1]
    def get_linear_dyn(self,x,u):
        cos_th = x[:, 0].unsqueeze(-1)
        sin_th = x[:, 1].unsqueeze(-1)
        dth = x[:, 2].unsqueeze(-1)
        dt = 0.05
        g, m, l = self.params.detach()
        D = self.batch_align([[sin_th * torch.sin(dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th,cos_th)) / (cos_th ** 2 + sin_th ** 2),
                -(cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.sin(dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),-dt * torch.sin(dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
                -3 * dt ** 2 * torch.sin(dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th,cos_th)) / (l ** 2 * m)
            ],
            [
                -sin_th * torch.cos(
                    dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th,
                                                                                                      cos_th)) / (
                            cos_th ** 2 + sin_th ** 2),
                (cos_th / (cos_th ** 2 + sin_th ** 2) + 3 * dt ** 2 * g / (2 * l)) * torch.cos(
                    dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
                dt * torch.cos(
                    dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th, cos_th)),
                3 * dt ** 2 * torch.cos(
                    dt * (dt * (3 * g * sin_th / (2 * l) + 3 * u / (l ** 2 * m)) + dth) + torch.atan2(sin_th,
                                                                                                      cos_th)) / (
                            l ** 2 * m)
            ],
            [
                0,
                3 * dt * g / (2 * l),
                1,
                3 * dt / (l ** 2 * m)
            ]
        ])
        return D
    def get_linear_dyn_test(self,x,u):
        cos_th = x[:, 0]
        sin_th = x[:, 1]
        dth = x[:, 2]
        u=u[:,0]
        dt = 0.05
        g, m, l = self.params.detach()
        D=torch.tensor([
            [
                sin_th * torch.sin(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)) / (
                            cos_th ** 2 + sin_th ** 2),
                -(cos_th / (cos_th ** 2 + sin_th ** 2) + 0.00375 * g / l) * torch.sin(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)),
                -0.05 * torch.sin(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)),
                -0.0075 * torch.sin(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)) / (
                            l ** 2 * m)
            ],
            [
                -sin_th * torch.cos(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)) / (
                            cos_th ** 2 + sin_th ** 2),
                (cos_th / (cos_th ** 2 + sin_th ** 2) + 0.00375 * g / l) * torch.cos(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)),
                0.05 * torch.cos(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)),
                0.0075 * torch.cos(
                    0.05 * dth + 0.00375 * g * sin_th / l + torch.atan2(sin_th, cos_th) + 0.0075 * u / (l ** 2 * m)) / (
                            l ** 2 * m)
            ],
            [
                0,
                0.075 * g / l,
                1,
                0.15 / (l ** 2 * m)
            ]
        ])
        return D
if __name__ == '__main__':
    dx = PendulumDx()
    batch, T = 50, 20
    U = torch.zeros(T, batch, dx.n_ctrl)
    X= torch.ones(T, batch, dx.n_state)
    # xinit = torch.zeros(n_batch, dx.n_state)
    # xinit[:,0] = np.cos(0)
    # xinit[:,1] = np.sin(0)
    # x = xinit
    g,m,l=dx.params.detach()
    #print(len(dx.params))
    time1=time.time()
    grad_D, grad_d, D_grad_x, D_grad_u, D, d_grad_X, d_grad_U=dx.grad_input(X,U)
    time2=time.time()
    import gym

    env = gym.make('Pendulum-v1')
    obs = env.reset()
    # env.state = np.array([np.pi, 0.])  # 设置自定义状态
    dx = PendulumDx()
    n_batch, T = 1, 20
    xinit = torch.zeros(n_batch, dx.n_state)
    uinit = torch.zeros(n_batch, dx.n_ctrl)
    # xinit[:,0] = np.cos(np.pi)
    # xinit[:,1] = np.sin(np.pi)

    xinit[:, 0] = torch.tensor(obs[0], dtype=torch.float32)  # cos(theta)
    xinit[:, 1] = torch.tensor(obs[1], dtype=torch.float32)  # sin(theta)
    xinit[:, 2] = torch.tensor(obs[2], dtype=torch.float32)

    xinit = torch.tensor([[np.cos(0.1),np.sin(0.1), 0.0]])  # 初始角度为 0.1 弧度
    uinit = torch.tensor([[0.0]])
    D=dx.get_linear_dyn_test(xinit,uinit)
    print(D)
    xinit.requires_grad=True
    uinit.requires_grad=True
    newx=dx(xinit,uinit)
    Rt,St=[],[]
    for j in range(3):
        Rj, Sj = torch.autograd.grad(
            newx[:, j].sum(), [xinit, uinit],
            retain_graph=True)
        Rt.append(Rj)
        St.append(Sj)
    Rt = torch.stack(Rt, dim=1)
    St = torch.stack(St, dim=1)
    print(Rt)
    #print(time2-time1)
    #dx.get_matrices(X,U)
    # for t in range(T):
    #     x = dx(x, u[t])
    #     fig, ax = dx.get_frame(x[0])
    #     fig.savefig('{:03d}.png'.format(t))
    #     plt.close(fig)
    #
    # vid_file = 'pendulum_vid.mp4'
    # if os.path.exists(vid_file):
    #     os.remove(vid_file)
    # cmd = ('(/usr/bin/ffmpeg -loglevel quiet '
    #         '-r 32 -f image2 -i %03d.png -vcodec '
    #         'libx264 -crf 25 -pix_fmt yuv420p {}/) &').format(
    #     vid_file
    # )
    # os.system(cmd)
    # for t in range(T):
    #     os.remove('{:03d}.png'.format(t))
