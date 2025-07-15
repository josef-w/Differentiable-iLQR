#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np
import sys
sys.path.append('C:\project\iLQR\mpc.pytorch.mine\mpc')
import util

import os

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class CartpoleDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 5
        self.n_ctrl = 1

        # model parameters
        if params is None:
            # gravity, masscart, masspole, length
            self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        assert len(self.params) == 4
        self.force_mag = 100.

        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_velocity = 10

        self.dt = 0.05

        self.lower = -self.force_mag
        self.upper = self.force_mag

        # 0  1      2        3   4
        # x dx cos(th) sin(th) dth
        self.goal_state = torch.Tensor(  [ 0.,  0.,  1., 0.,   0.])
        self.goal_weights = torch.Tensor([0.1, 0.1,  1., 1., 0.1])
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        u = torch.clamp(u[:,0], -self.force_mag, self.force_mag)

        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        th = torch.atan2(sin_th, cos_th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        th_acc = (gravity * sin_th - cos_th * cart_in) / \
                 (length * (4./3. - masspole * cos_th**2 /
                                     total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        state = torch.stack((
            x, dx, torch.cos(th), torch.sin(th), dth
        ), 1)

        return state

    def batch_stack(self,matrix_n_part_n):
        # 按照列表stack 其中的tensor，得到6*10*1的tensor，然后再stack为5*6*10*1的tensor，再permute
        return torch.stack([torch.stack(
            [item for item in matrix_n_part_n_1]) for matrix_n_part_n_1 in matrix_n_part_n]).squeeze(-1).permute(2, 0,
                                                                                                                 1)

    def get_matrices(self,x, u):
        batch_size = x.shape[0]
        cos_th = x[:, 2].unsqueeze(-1)
        sin_th = x[:, 3].unsqueeze(-1)
        dth = x[:, 4].unsqueeze(-1)
        dt = self.dt
        g, m_c, m_p, l = torch.unbind(self.params.detach())

        ####矩阵1 5x6
        # 思路，注意到很多列表使用标量和张量混合，首先将其统一为张量的shape
        # 首先更改标量的形状为cos_th的形状，现在每个子列表中有6个元素，每个元素的shape为10*1，因此现在的列表形状为5*6*10*1

        D = self.batch_stack([
            [torch.ones_like(cos_th), torch.ones_like(cos_th) * dt, torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             torch.zeros_like(cos_th), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.ones_like(cos_th), dt * (-9 * cos_th ** 2 * m_p ** 2 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (8 * (m_c + m_p) ** 2 * (
                    -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + cos_th * m_p * (
                                                                              dth ** 2 * l * m_p * sin_th + u) / (
                                                                              (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (
                                                                                  m_c + m_p) + 4 / 3)) - m_p * (
                                                                              -cos_th * (
                                                                                  dth ** 2 * l * m_p * sin_th + u) / (
                                                                                          m_c + m_p) + g * sin_th) / (
                                                                              (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                  m_c + m_p) + 4 / 3))),
             dt * (-cos_th * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * l * m_p / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * l * m_p * sin_th / (
                           m_c + m_p)),
             dt * (cos_th ** 2 * m_p / ((m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1 / (
                         m_c + m_p))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -dt * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             cos_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             dt * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * (
                     dth ** 2 * l * m_p * sin_th + u) / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th * dt * dth * m_p * sin_th / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1,
             -cos_th * dt / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        # 时间为0.002秒
        # end = time.time()-start_time
        # print(end)

        # 定义矩阵 2 (3维张量: 5x6x4)
        # 定义矩阵 2的三个部分 (分别是5x6的矩阵)
        # 第一部分的 matrix 转换
        matrix_2_part_1 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), dt * (-9 * cos_th ** 2 * m_p ** 2 * sin_th / (
                    8 * (m_c + m_p) ** 2 * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - m_p * sin_th / (
                                                                               (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) + 4 / 3))),
             -cos_th * dt * m_p / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)), torch.zeros_like(cos_th),
             torch.zeros_like(cos_th)],
            torch.zeros((6, batch_size, 1)),
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * sin_th / (
                         8 * l * (m_c + m_p) * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2),
             dt / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)), torch.zeros_like(cos_th), torch.zeros_like(cos_th)]
        ])
        # print(matrix_2_part_1.shape)
        # 第二部分的 matrix 转换
        matrix_2_part_2 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), dt * (27 * cos_th ** 4 * m_p ** 3 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (16 * (m_c + m_p) ** 4 * (
                    -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 3) - 9 * cos_th ** 3 * m_p ** 2 * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                                   8 * (m_c + m_p) ** 4 * (
                                                                                   -3 * cos_th ** 2 * m_p / (4 * (
                                                                                       m_c + m_p)) + 1) ** 2) - cos_th ** 3 * m_p ** 2 * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                                   (m_c + m_p) ** 4 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3) ** 2) + 9 * cos_th ** 2 * m_p ** 2 * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               4 * (m_c + m_p) ** 3 * (
                                                                               -3 * cos_th ** 2 * m_p / (4 * (
                                                                                   m_c + m_p)) + 1) ** 2) + cos_th ** 2 * m_p ** 2 * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               (m_c + m_p) ** 3 * (
                                                                               -cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) + 4 / 3) ** 2) - 3 * cos_th * m_p * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                               (m_c + m_p) ** 3 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3)) + m_p * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3))),
             dt * (cos_th ** 3 * m_p ** 2 * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / ((m_c + m_p) ** 3 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - cos_th ** 2 * dth ** 2 * l * m_p ** 2 / (
                           (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + cos_th * m_p * (
                           -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - dth ** 2 * l * m_p / (
                           m_c + m_p) ** 2),
             dt * (-2 * cos_th ** 4 * dth * l * m_p ** 3 * sin_th / ((m_c + m_p) ** 4 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 4 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th / (
                           (m_c + m_p) ** 3 * (
                           -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - 2 * dth * l * m_p * sin_th / (
                           m_c + m_p) ** 2),
             dt * (-cos_th ** 4 * m_p ** 2 / (
                     (m_c + m_p) ** 4 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 2 * cos_th ** 2 * m_p / (
                           (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - 1 / (m_c + m_p) ** 2)],
            torch.zeros((6, batch_size, 1)),
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), -27 * cos_th ** 3 * dt * m_p ** 2 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     16 * l * (m_c + m_p) ** 3 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 3) + 9 * cos_th ** 2 * dt * m_p * (
                     dth ** 2 * l * m_p * sin_th + u) / (8 * l * (m_c + m_p) ** 3 * (
                    -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + cos_th ** 2 * dt * m_p * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l * (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 9 * cos_th * dt * m_p * (
                     -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) ** 2 * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + dt * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (-cos_th ** 2 * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (l * (m_c + m_p) ** 2 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + cos_th * dt * dth ** 2 * m_p / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             2 * cos_th ** 3 * dt * dth * m_p ** 2 * sin_th / ((m_c + m_p) ** 3 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + 2 * cos_th * dt * dth * m_p * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             cos_th ** 3 * dt * m_p / (
                     l * (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + cos_th * dt / (
                     l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        # print(matrix_2_part_2.shape)

        # 第三部分的 matrix 转换
        matrix_2_part_3 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), dt * (-9 * cos_th ** 2 * m_p ** 2 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) * (
                                                                               -3 * cos_th ** 2 * m_p / (2 * (
                                                                                   m_c + m_p) ** 2) + 3 * cos_th ** 2 / (
                                                                                       2 * (m_c + m_p))) / (
                                                                                   8 * (m_c + m_p) ** 2 * (
                                                                                   -3 * cos_th ** 2 * m_p / (4 * (
                                                                                       m_c + m_p)) + 1) ** 3) - 9 * cos_th ** 2 * m_p ** 2 * (
                                                                               -cos_th * dth ** 2 * l * sin_th / (
                                                                                   m_c + m_p) + cos_th * (
                                                                                       dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) ** 2) / (
                                                                                   8 * (m_c + m_p) ** 2 * (
                                                                                   -3 * cos_th ** 2 * m_p / (4 * (
                                                                                       m_c + m_p)) + 1) ** 2) + 9 * cos_th ** 2 * m_p ** 2 * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               4 * (m_c + m_p) ** 3 * (
                                                                               -3 * cos_th ** 2 * m_p / (4 * (
                                                                                   m_c + m_p)) + 1) ** 2) - 9 * cos_th ** 2 * m_p * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               4 * (m_c + m_p) ** 2 * (
                                                                                   -3 * cos_th ** 2 * m_p / (
                                                                                   4 * (
                                                                                       m_c + m_p)) + 1) ** 2) + cos_th * dth ** 2 * l * m_p * sin_th / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3)) + cos_th * m_p * (
                                                                               -cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) ** 2 + cos_th ** 2 / (
                                                                                           m_c + m_p)) * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                                   (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3) ** 2) - 2 * cos_th * m_p * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                               (m_c + m_p) ** 3 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3)) + cos_th * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3)) - m_p * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) * (
                                                                               -cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) ** 2 + cos_th ** 2 / (
                                                                                           m_c + m_p)) / (
                                                                               (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) + 4 / 3) ** 2) - m_p * (
                                                                               -cos_th * dth ** 2 * l * sin_th / (
                                                                                   m_c + m_p) + cos_th * (
                                                                                       dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) ** 2) / (
                                                                               (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) + 4 / 3)) + m_p * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3)) - (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                   m_c + m_p) + 4 / 3))),
             dt * (-cos_th * m_p * (-cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) * (
                     -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - cos_th * m_p * (
                           cos_th * dth ** 2 * l * m_p / (m_c + m_p) ** 2 - cos_th * dth ** 2 * l / (m_c + m_p)) / (
                           (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + cos_th * m_p * (
                           -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - cos_th * (
                           -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - dth ** 2 * l * m_p / (
                           m_c + m_p) ** 2 + dth ** 2 * l / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th * (
                     -cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) / ((m_c + m_p) ** 2 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 4 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th / (
                           (m_c + m_p) ** 3 * (
                           -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 4 * cos_th ** 2 * dth * l * m_p * sin_th / (
                           (m_c + m_p) ** 2 * (
                           -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - 2 * dth * l * m_p * sin_th / (
                           m_c + m_p) ** 2 + 2 * dth * l * sin_th / (m_c + m_p)),
             dt * (cos_th ** 2 * m_p * (-cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 2 * cos_th ** 2 * m_p / (
                           (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + cos_th ** 2 / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - 1 / (m_c + m_p) ** 2)],
            torch.zeros((6, batch_size, 1)),
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) * (
                     -3 * cos_th ** 2 * m_p / (2 * (m_c + m_p) ** 2) + 3 * cos_th ** 2 / (2 * (m_c + m_p))) / (
                     8 * l * (m_c + m_p) * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 3) + 9 * cos_th * dt * m_p * (
                     -cos_th * dth ** 2 * l * sin_th / (m_c + m_p) + cos_th * (dth ** 2 * l * m_p * sin_th + u) / (
                     m_c + m_p) ** 2) / (8 * l * (m_c + m_p) * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - 9 * cos_th * dt * m_p * (
                     -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + 9 * cos_th * dt * (
                     -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (8 * l * (m_c + m_p) * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * dth ** 2 * sin_th / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - dt * (
                     -cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + dt * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (-cos_th ** 2 * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + cos_th * dt * dth ** 2 * l * m_p / (
                           m_c + m_p) ** 2),
             -2 * cos_th * dt * dth * m_p * sin_th * (
                         -cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) / (
                     (m_c + m_p) * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + 2 * cos_th * dt * dth * m_p * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             cos_th * dt / (l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        # print(matrix_2_part_3.shape)

        # 第四部分的 matrix 转换
        matrix_2_part_4 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             dt * (9 * cos_th ** 3 * dth ** 2 * m_p ** 3 * sin_th / (8 * (m_c + m_p) ** 3 * (
                     -3 * cos_th ** 2 * m_p / (
                     4 * (m_c + m_p)) + 1) ** 2) + 2 * cos_th * dth ** 2 * m_p ** 2 * sin_th / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             dt * (cos_th ** 2 * dth ** 2 * m_p ** 2 / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * m_p / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * dth * m_p ** 2 * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * m_p * sin_th / (
                           m_c + m_p)), torch.zeros_like(cos_th)],
            torch.zeros((6, batch_size, 1)),
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -9 * cos_th ** 2 * dt * dth ** 2 * m_p ** 2 * sin_th / (8 * l * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - 9 * cos_th * dt * m_p * (
                     -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l ** 2 * (m_c + m_p) * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * dth ** 2 * m_p * sin_th / (
                     l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dt * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l ** 2 * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -cos_th * dt * dth ** 2 * m_p / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - dt * (
                     -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     l ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)), torch.zeros_like(cos_th),
             cos_th * dt / (l ** 2 * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])

        # 合并三个部分为 batch*5x6x4 的张量   5x6x4
        D_grad_params = torch.stack([matrix_2_part_1, matrix_2_part_2, matrix_2_part_3, matrix_2_part_4], dim=3)
        # print(D_grad_params.shape)

        # 定义矩阵 3 (3维张量: 5x6x5)
        # 第一个 matrix 转换
        # matrix_3_part_1 = torch.tensor([
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0]
        # ])
        matrix_3_part_1 = torch.zeros((batch_size, 5, 6))

        # 第二个 matrix 转换
        # matrix_3_part_2 = torch.tensor([
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0]
        # ])
        matrix_3_part_2 = torch.zeros((batch_size, 5, 6))

        # 第三个 matrix 转换
        matrix_3_part_3 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), dt * (-27 * cos_th ** 3 * m_p ** 3 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (8 * (m_c + m_p) ** 3 * (
                    -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 3) + 9 * cos_th ** 2 * m_p ** 2 * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                                   8 * (m_c + m_p) ** 3 * (
                                                                                   -3 * cos_th ** 2 * m_p / (4 * (
                                                                                       m_c + m_p)) + 1) ** 2) + 2 * cos_th ** 2 * m_p ** 2 * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                                   (m_c + m_p) ** 3 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3) ** 2) - 9 * cos_th * m_p ** 2 * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               4 * (m_c + m_p) ** 2 * (
                                                                               -3 * cos_th ** 2 * m_p / (4 * (
                                                                                   m_c + m_p)) + 1) ** 2) - 2 * cos_th * m_p ** 2 * (
                                                                               -cos_th * (
                                                                                   dth ** 2 * l * m_p * sin_th + u) / (
                                                                                           m_c + m_p) + g * sin_th) / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3) ** 2) + 2 * m_p * (
                                                                               dth ** 2 * l * m_p * sin_th + u) / (
                                                                               (m_c + m_p) ** 2 * (
                                                                                   -cos_th ** 2 * m_p / (
                                                                                       m_c + m_p) + 4 / 3))),
             dt * (-2 * cos_th ** 2 * m_p ** 2 * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                         (m_c + m_p) ** 2 * (
                         -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + cos_th * dth ** 2 * l * m_p ** 2 / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - m_p * (
                           -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             dt * (4 * cos_th ** 3 * dth * l * m_p ** 3 * sin_th / ((m_c + m_p) ** 3 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + 4 * cos_th * dth * l * m_p ** 2 * sin_th / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             dt * (2 * cos_th ** 3 * m_p ** 2 / (
                     (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + 2 * cos_th * m_p / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -2 * cos_th * sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 - sin_th ** 2 * torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2,
             2 * cos_th ** 2 * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 + cos_th * sin_th * torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2 - torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             dt * sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             2 * cos_th * sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 - sin_th ** 2 * torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2,
             -2 * cos_th ** 2 * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 + cos_th * sin_th * torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2 + torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             dt * sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             27 * cos_th ** 2 * dt * m_p ** 2 * (
                         -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 3) - 9 * cos_th * dt * m_p * (
                     dth ** 2 * l * m_p * sin_th + u) / (8 * l * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - 2 * cos_th * dt * m_p * (
                     dth ** 2 * l * m_p * sin_th + u) / (
                     l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) + 9 * dt * m_p * (
                     -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2),
             2 * cos_th * dt * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - dt * dth ** 2 * m_p / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -4 * cos_th ** 2 * dt * dth * m_p ** 2 * sin_th / ((m_c + m_p) ** 2 * (
                     -cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - 2 * dt * dth * m_p * sin_th / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th ** 2 * dt * m_p / (
                         l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3) ** 2) - dt / (
                     l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])

        # 第四个 matrix 转换
        matrix_3_part_4 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             dt * (-9 * cos_th ** 2 * m_p ** 2 * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     8 * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + cos_th * dth ** 2 * l * m_p ** 2 / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - m_p * (
                           -cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                           (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             torch.zeros_like(cos_th), dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 / (
                    (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * l * m_p / (m_c + m_p)),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             cos_th * sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 - 2 * sin_th ** 2 * torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2 + torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th ** 2 * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 + 2 * cos_th * sin_th * torch.sin(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2,
             -cos_th * dt * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             cos_th * sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 + 2 * sin_th ** 2 * torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2 - torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th ** 2 * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (
                     cos_th ** 2 + sin_th ** 2) ** 2 - 2 * cos_th * sin_th * torch.cos(
                 dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2) ** 2,
             -cos_th * dt * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (8 * l * (m_c + m_p) * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * dth ** 2 * m_p / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             torch.zeros_like(cos_th),
             -2 * cos_th * dt * dth * m_p / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             torch.zeros_like(cos_th)]
        ])

        # 第五个 matrix 转换
        matrix_3_part_5 = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             dt * (9 * cos_th ** 3 * dth * l * m_p ** 3 * sin_th / (4 * (m_c + m_p) ** 3 * (
                     -3 * cos_th ** 2 * m_p / (
                         4 * (m_c + m_p)) + 1) ** 2) + 4 * cos_th * dth * l * m_p ** 2 * sin_th / (
                           (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))),
             dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * l * m_p / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * l * m_p ** 2 * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * l * m_p * sin_th / (
                           m_c + m_p)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             dt * sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th * dt * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -dt ** 2 * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             dt * sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th * dt * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -dt ** 2 * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -9 * cos_th ** 2 * dt * dth * m_p ** 2 * sin_th / (4 * (m_c + m_p) ** 2 * (
                     -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - 2 * dt * dth * m_p * sin_th / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th * dt * dth * m_p / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th * dt * m_p * sin_th / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             torch.zeros_like(cos_th)]
        ])

        # 合并三个部分为 batch*5x6x5 的张量
        D_grad_x = torch.stack([matrix_3_part_1, matrix_3_part_2, matrix_3_part_3, matrix_3_part_4, matrix_3_part_5],
                               dim=3)
        # print(D_grad_x.shape)
        # 定义矩阵 4 batch*5x6x1
        D_grad_u = self.batch_stack([
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), dt * (9 * cos_th ** 3 * m_p ** 2 / (
                        8 * (m_c + m_p) ** 3 * (
                            -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + 2 * cos_th * m_p / (
                                                                                   (m_c + m_p) ** 2 * (
                                                                                       -cos_th ** 2 * m_p / (
                                                                                           m_c + m_p) + 4 / 3))),
             torch.zeros_like(cos_th), torch.zeros_like(cos_th), torch.zeros_like(cos_th)],
            torch.zeros((6, batch_size, 1)),
            torch.zeros((6, batch_size, 1)),
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th), -9 * cos_th ** 2 * dt * m_p / (
                        8 * l * (m_c + m_p) ** 2 * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt / (
                         l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)), torch.zeros_like(cos_th),
             torch.zeros_like(cos_th), torch.zeros_like(cos_th)]
        ]).unsqueeze(-1)
        # print(D_grad_u[0])

        # dxt/dtheta(5x4)

        x_grad_theta = self.batch_stack([
            torch.zeros((4, batch_size, 1)),
            [-cos_th * dt * m_p * sin_th / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (9 * cos_th ** 3 * m_p ** 2 * (
                         -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                               16 * (m_c + m_p) ** 3 * (
                                   -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - cos_th ** 2 * m_p * (
                               dth ** 2 * l * m_p * sin_th + u) / (
                               (m_c + m_p) ** 3 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + cos_th * m_p * (
                               -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                               (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - (
                               dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) ** 2),
             dt * (-9 * cos_th * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) * (
                         -cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) / (16 * (m_c + m_p) * (
                         -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - cos_th * m_p * (
                               -cos_th * dth ** 2 * l * sin_th / (m_c + m_p) + cos_th * (
                                   dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) ** 2) / (
                               (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + cos_th * m_p * (
                               -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                               (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - cos_th * (
                               -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                               (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * l * sin_th / (
                               m_c + m_p) - (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) ** 2),
             dt * (cos_th ** 2 * dth ** 2 * m_p ** 2 * sin_th / (
                         (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * m_p * sin_th / (
                               m_c + m_p))],
            torch.zeros((4, batch_size, 1)),
            torch.zeros((4, batch_size, 1)),
            [dt * sin_th / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -9 * cos_th ** 2 * dt * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                         16 * l * (m_c + m_p) ** 2 * (
                             -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + cos_th * dt * (
                         dth ** 2 * l * m_p * sin_th + u) / (
                         l * (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             9 * dt * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) * (
                         -cos_th ** 2 * m_p / (m_c + m_p) ** 2 + cos_th ** 2 / (m_c + m_p)) / (
                         16 * l * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + dt * (
                         -cos_th * dth ** 2 * l * sin_th / (m_c + m_p) + cos_th * (dth ** 2 * l * m_p * sin_th + u) / (
                             m_c + m_p) ** 2) / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -cos_th * dt * dth ** 2 * m_p * sin_th / (
                         l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) - dt * (
                         -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                         l ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        # print(x_grad_theta.shape)

        # dx/dxt-1 (5x5)
        x_grad_xtm1 = self.batch_stack([
            [torch.zeros_like(cos_th), dt * torch.ones_like(cos_th), torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.ones_like(cos_th), dt * (-9 * cos_th ** 2 * m_p ** 2 * (
                        -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                                                                                  8 * (m_c + m_p) ** 2 * (
                                                                                      -3 * cos_th ** 2 * m_p / (4 * (
                                                                                          m_c + m_p)) + 1) ** 2) + cos_th * m_p * (
                                                                                  dth ** 2 * l * m_p * sin_th + u) / (
                                                                                  (m_c + m_p) ** 2 * (
                                                                                      -cos_th ** 2 * m_p / (
                                                                                          m_c + m_p) + 4 / 3)) - m_p * (
                                                                                  -cos_th * (
                                                                                      dth ** 2 * l * m_p * sin_th + u) / (
                                                                                              m_c + m_p) + g * sin_th) / (
                                                                                  (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                                      m_c + m_p) + 4 / 3))),
             dt * (-cos_th * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                         (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * l * m_p / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th / (
                         (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * l * m_p * sin_th / (
                               m_c + m_p))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -dt * torch.sin(dt * dth + torch.atan2(sin_th, cos_th))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             cos_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             dt * torch.cos(dt * dth + torch.atan2(sin_th, cos_th))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                         8 * l * (m_c + m_p) * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * (
                         dth ** 2 * l * m_p * sin_th + u) / (
                         l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th * dt * dth * m_p * sin_th / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1]
        ])
        # print(x_grad_xtm1.shape)

        # dxt/dut-1 (5x1)
        x_grad_utm1 = self.batch_stack([
            [torch.zeros_like(cos_th)],
            [dt * (cos_th ** 2 * m_p / ((m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1 / (
                        m_c + m_p))],
            [torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th)],
            [-cos_th * dt / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        # print(x_grad_utm1.shape)
        # print(x_grad_utm1[0])
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
        dx, du = n_state, n_ctrl
        dtotal = dx + du
        dtheta = 4
        # 初始化输出列表 grad_D 和 grad_d
        grad_D = []
        grad_d = []

        _X, _U = X.reshape(T * batch_size, dx), U.reshape(T * batch_size, du)
        D, D_grad_params, D_grad_x, D_grad_u, x_grad_theta, x_grad_xtm1, x_grad_utm1 = self.get_matrices(_X, _U)
        D, D_grad_params, D_grad_x, D_grad_u = D.reshape(T, batch_size, dx, dtotal), D_grad_params.reshape(T,
                                                                                                           batch_size,
                                                                                                           dx, dtotal,
                                                                                                           dtheta), D_grad_x.reshape(
            T, batch_size, dx, dtotal, dx), D_grad_u.reshape(T, batch_size, dx, dtotal, du)
        x_grad_theta, x_grad_xtm1, x_grad_utm1 = x_grad_theta.reshape(T, batch_size, dx, dtheta), x_grad_xtm1.reshape(T,
                                                                                                                      batch_size,
                                                                                                                      dx,
                                                                                                                      dx), x_grad_utm1.reshape(
            T, batch_size, dx, du)
        # 初始 gradxt 设为全零张量
        gradxt = torch.zeros(batch_size, n_state, dtheta)
        X_U = torch.cat((X, U), dim=-1)
        d_grad_X = torch.einsum('tbnmk,tbm->tbnk', -D_grad_x, X_U)
        d_grad_U = torch.einsum('tbnmk,tbm->tbnk', -D_grad_u, X_U)

        for t in range(T):
            # 设置示例张量
            if K == None:
                ut_grad_xt = torch.zeros(batch_size, n_ctrl, n_state)  ####K_t
                utm1_grad_xtm1 = torch.zeros(batch_size, n_ctrl, n_state)  ####K_tm1
            else:
                ut_grad_xt = K[t]
                if t > 0:
                    utm1_grad_xtm1 = K[t - 1]

            # 计算 gradxt
            if t > 0:
                gradxtm1 = gradxt
                gradxt = x_grad_theta[t] + torch.matmul(x_grad_xtm1[t] + torch.matmul(x_grad_utm1[t], utm1_grad_xtm1),
                                                        gradxt)

            # 计算 gradDt
            if t < (T - 1):
                gradDt = D_grad_params[t] + torch.matmul(
                    D_grad_x[t] + torch.matmul(D_grad_u[t], ut_grad_xt.unsqueeze(1)), gradxt.unsqueeze(1))
                grad_D.append(gradDt)
            if t > 0:
                ####in the t+1 step, trace back the t step
                xtm1_utm1_params = torch.cat((gradxtm1, torch.matmul(utm1_grad_xtm1, gradxtm1)), dim=1)
                xtm1_utm1 = torch.cat((X[t - 1], U[t - 1]), dim=-1)
                grad_dtm1 = gradxt - torch.einsum('bnmk,bm->bnk', grad_D[t - 1], xtm1_utm1) - torch.matmul(D[t - 1],
                                                                                                           xtm1_utm1_params)
                grad_d.append(grad_dtm1)

        # 在循环结束后，将列表转换为 tensor
        grad_D = torch.stack(grad_D, dim=0)
        grad_d = torch.stack(grad_d, dim=0)

        return grad_D, grad_d, D_grad_x[:T - 1], D_grad_u[:T - 1], D[:T - 1], d_grad_X[:T - 1], d_grad_U[:T - 1]

    def get_linear_dyn(self, x, u):
        batch_size = x.shape[0]
        cos_th = x[:, 2].unsqueeze(-1)
        sin_th = x[:, 3].unsqueeze(-1)
        dth = x[:, 4].unsqueeze(-1)
        dt = self.dt
        g, m_c, m_p, l = torch.unbind(self.params.detach())

        ####矩阵1 5x6
        # 思路，注意到很多列表使用标量和张量混合，首先将其统一为张量的shape
        # 首先更改标量的形状为cos_th的形状，现在每个子列表中有6个元素，每个元素的shape为10*1，因此现在的列表形状为5*6*10*1

        D = self.batch_stack([
            [torch.ones_like(cos_th), torch.ones_like(cos_th) * dt, torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             torch.zeros_like(cos_th), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.ones_like(cos_th), dt * (-9 * cos_th ** 2 * m_p ** 2 * (
                    -cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (8 * (m_c + m_p) ** 2 * (
                    -3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) + cos_th * m_p * (
                                                                              dth ** 2 * l * m_p * sin_th + u) / (
                                                                              (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (
                                                                              m_c + m_p) + 4 / 3)) - m_p * (
                                                                              -cos_th * (
                                                                              dth ** 2 * l * m_p * sin_th + u) / (
                                                                                      m_c + m_p) + g * sin_th) / (
                                                                              (m_c + m_p) * (-cos_th ** 2 * m_p / (
                                                                              m_c + m_p) + 4 / 3))),
             dt * (-cos_th * m_p * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (
                     (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + dth ** 2 * l * m_p / (m_c + m_p)),
             dt * (2 * cos_th ** 2 * dth * l * m_p ** 2 * sin_th / (
                     (m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 2 * dth * l * m_p * sin_th / (
                           m_c + m_p)),
             dt * (cos_th ** 2 * m_p / ((m_c + m_p) ** 2 * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1 / (
                     m_c + m_p))],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             sin_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -cos_th * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             -dt * torch.sin(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             -sin_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             cos_th * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)) / (cos_th ** 2 + sin_th ** 2),
             dt * torch.cos(dt * dth + torch.atan2(sin_th, cos_th)), torch.zeros_like(cos_th)],
            [torch.zeros_like(cos_th), torch.zeros_like(cos_th),
             9 * cos_th * dt * m_p * (-cos_th * (dth ** 2 * l * m_p * sin_th + u) / (m_c + m_p) + g * sin_th) / (
                     8 * l * (m_c + m_p) * (-3 * cos_th ** 2 * m_p / (4 * (m_c + m_p)) + 1) ** 2) - dt * (
                     dth ** 2 * l * m_p * sin_th + u) / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             dt * (-cos_th * dth ** 2 * l * m_p / (m_c + m_p) + g) / (l * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)),
             -2 * cos_th * dt * dth * m_p * sin_th / ((m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3)) + 1,
             -cos_th * dt / (l * (m_c + m_p) * (-cos_th ** 2 * m_p / (m_c + m_p) + 4 / 3))]
        ])
        return D

    def get_frame(self, state, ax=None):
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 5
        x, dx, cos_th, sin_th, dth = torch.unbind(state)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th*length
        th_y = cos_th*length

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        ax.plot((x,x+th_x), (0, th_y), color='k')
        ax.set_xlim((-length*2, length*2))
        ax.set_ylim((-length*2, length*2))
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

if __name__ == '__main__':
    dx = CartpoleDx()
    n_batch, T = 2, 10
    u = torch.randn(T, n_batch, dx.n_ctrl)
    x = torch.randn(T, n_batch, dx.n_state)
    xinit = torch.zeros(n_batch, dx.n_state)
    th = 1.
    xinit[:,2] = np.cos(th)
    xinit[:,3] = np.sin(th)
    # x = xinit
    dx.grad_input(x,u)
    # for t in range(T):
    #     x = dx(x, u[t])
    #     fig, ax = dx.get_frame(x[0])
    #     fig.savefig('{:03d}.png'.format(t))
    #     plt.close(fig)
    #
    # vid_file = 'cartpole_vid.mp4'
    # if os.path.exists(vid_file):
    #     os.remove(vid_file)
    # cmd = ('{} -loglevel quiet '
    #         '-r 32 -f image2 -i %03d.png -vcodec '
    #         'libx264 -crf 25 -pix_fmt yuv420p {}').format(
    #     FFMPEG_BIN,
    #     vid_file
    # )
    # os.system(cmd)
    # for t in range(T):
    #     os.remove('{:03d}.png'.format(t))
