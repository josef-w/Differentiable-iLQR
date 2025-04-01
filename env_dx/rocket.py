import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from casadi import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import numpy as np
import sys
import os
import time

class RocketDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        
        # System parameters
        self.dt = 0.1  # Time step
        self.n_state = 13  # State dimension: [r(3), v(3), q(4), w(3)]
        self.n_ctrl = 3    # Control dimension: [f, f_side(2)]
        
        # System parameter initialization
        if params is None:
            # Jx, Jy, Jz, mass, l
            self.params = torch.tensor((0.5, 1.0, 1.0, 1.0, 1.0), requires_grad=True)
        else:
            self.params = params
        #'wr': 10, 'wv': 1, 'wtilt': 50, 'ww': 1, 'wsidethrust': 1, 'wthrust': 0.4, 'max_f_sq': 400, 'max_tilt_angle': 0.3, 'dt': 0.1,
        # Target state
        self.goal_state = torch.zeros(self.n_state)
        # Target position: vertical hover
        self.goal_state[2] = 0.0  # Target height
        # Target velocity: stationary
        self.goal_state[3:6] = 0.0
        # Target attitude: vertical upward [1,0,0,0]
        self.goal_state[6] = 1.0
        # Target angular velocity: stationary
        self.goal_state[10:] = 0.0
        
        # Weights
        self.goal_weights = torch.ones(self.n_state)
        # Position weights [1,1,1]
        self.goal_weights[0:3] = 10.0
        # Velocity weights [1,1,1]
        self.goal_weights[3:6] = 1.0
        # Quaternion attitude weights [0.1,0.1,0.1,0.1]
        self.goal_weights[6:10] = 0.1
        # Angular velocity weights [1,1,1]
        self.goal_weights[10:] = 1.0

        # Control penalty weights
        self.side_penalty = 1  # Side thrust weight
        self.thrust_penalty = 0.4  # Main thrust weight
        self.ctrl_penalty=torch.tensor([self.side_penalty, self.side_penalty, self.thrust_penalty])
        
        # Tilt angle weight
        self.tilt_penalty = 50.0
        
        # Constraints
        self.max_thrust = 20**2
        self.max_tilt_angle = 0.3
        
        # MPC parameters
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5
        
        # Calculate quadratic approximation of tilt angle at initialization
        # self.tilt_Q, self.tilt_p = self.get_tilt_angle_quadratic(self.goal_state)
        self.tilt_Q=torch.tensor([0., 0., 4., 4.])
        self.tilt_p=torch.tensor([0., 0., 0., 0.])
        # Multiply Q and p by weight
        self.tilt_Q = self.tilt_penalty * self.tilt_Q
        self.tilt_p = self.tilt_penalty * self.tilt_p

        self.lower, self.upper = torch.tensor([-20., -20., -20.0]) ,torch.tensor([20., 20., 20.])

    def forward(self, x, u):
        """
        Fully debugged batch version, strictly maintaining the physical model of forward2
        Input:
            x: [batch_size, 13] or [13]
            u: [batch_size, 3] or [3]
        Output:
            new_x: new state with same dimensions as input x
        """
        # Dimension handling for batch computation
        input_was_1d = x.ndim == 1
        if input_was_1d:
            x = x.unsqueeze(0)  # [1, 13]
            u = u.unsqueeze(0)  # [1, 3]

        batch_size = x.size(0)

        # 1. Parameter unpacking (identical to forward2)
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        J = torch.diag(torch.tensor([Jx, Jy, Jz], device=x.device))
        J_batch = J.unsqueeze(0).expand(batch_size, -1, -1)  # [b,3,3]

        # 2. State unpacking
        v = x[:, 3:6]  # [b,3]
        q = x[:, 6:10]  # [b,4]
        w = x[:, 10:13]  # [b,3]

        # 3. Control input processing (maintain [f, f_side1, f_side2] order)
        T_B = u.clone()  # [b,3]
        T_B = torch.clamp(T_B, -self.max_thrust, self.max_thrust)

        # 4. Direction cosine matrix (exactly matching forward2)
        q0, q1, q2, q3 = q.unbind(-1)
        # Efficient batch computation of C_B_I
        C_B_I = torch.stack([
            torch.stack([1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)], dim=-1),
            torch.stack([2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 + q0 * q1)], dim=-1),
            torch.stack([2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)], dim=-1)
        ], dim=1)  # [b,3,3]
        C_I_B = C_B_I.transpose(1, 2)  # [b,3,3]

        # 5. Dynamics computation (strictly corresponding to forward2)
        # Gravity
        g = torch.tensor([-10., 0., 0.], device=x.device).expand(batch_size, 3)

        # Position derivative
        dr_I = v

        # Velocity derivative (fixed bmm dimension issue)
        thrust_global = torch.bmm(C_I_B, T_B.unsqueeze(-1)).squeeze(-1)  # [b,3,3] @ [b,3,1] -> [b,3,1] -> [b,3]
        dv_I = thrust_global / mass + g

        # Quaternion derivative (optimized batch computation)
        omega = torch.zeros(batch_size, 4, 4, device=x.device)
        omega[:, 0, 1:] = -w
        omega[:, 1:, 0] = w
        omega[:, 1, 2] = w[:, 2]
        omega[:, 1, 3] = -w[:, 1]
        omega[:, 2, 1] = -w[:, 2]
        omega[:, 2, 3] = w[:, 0]
        omega[:, 3, 1] = w[:, 1]
        omega[:, 3, 2] = -w[:, 0]
        dq = 0.5 * torch.bmm(omega, q.unsqueeze(-1)).squeeze(-1)

        # Angular velocity derivative
        r_T_B = torch.tensor([-l / 2, 0., 0.], device=x.device).expand(batch_size, 3)
        torque = torch.cross(r_T_B, T_B)

        Jw = torch.bmm(J_batch, w.unsqueeze(-1)).squeeze(-1)
        w_cross_Jw = torch.cross(w, Jw)
        dw = torch.bmm(torch.inverse(J_batch), (torque - w_cross_Jw).unsqueeze(-1)).squeeze(-1)

        # 6. State update
        derivatives = torch.cat([dr_I, dv_I, dq, dw], dim=1)
        new_x = x + derivatives * self.dt

        # Quaternion normalization
        new_q = new_x[:, 6:10]
        normed_q = new_q / (new_q.norm(dim=1, keepdim=True) + 1e-8)
        new_x_out = new_x.clone()
        new_x_out[:, 6:10] = normed_q

        return new_x.squeeze(0) if input_was_1d else new_x

    def quaternion_to_rotation(self, q):
        """Convert quaternion to rotation matrix
        Args:
            q: (batch_size, 4) quaternion [w, x, y, z]
        Returns:
            R: (batch_size, 3, 3) rotation matrix
        """
        w, x, y, z = torch.unbind(q, dim=-1)
        R = torch.stack([
            torch.stack([1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y], dim=-1),
            torch.stack([2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x], dim=-1),
            torch.stack([2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y], dim=-1)
        ], dim=-1)
        return R

    def quaternion_mul(self, p, q):
        """Quaternion multiplication"""
        p0, p1, p2, p3 = torch.unbind(p, dim=-1)
        q0, q1, q2, q3 = torch.unbind(q, dim=-1)
        
        return torch.stack([
            p0*q0 - p1*q1 - p2*q2 - p3*q3,
            p0*q1 + p1*q0 + p2*q3 - p3*q2,
            p0*q2 - p1*q3 + p2*q0 + p3*q1,
            p0*q3 + p1*q2 - p2*q1 + p3*q0
        ], dim=-1)

    def get_tilt_angle(self, q):
        """Calculate tilt angle
        Args:
            q: (batch_size, 4) quaternion [w, x, y, z]
        Returns:
            theta: (batch_size,) tilt angle
        """
        # Ensure quaternion is unit quaternion
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        
        # Extract rotation matrix from quaternion
        R = self.quaternion_to_rotation(q)
        # Calculate angle with vertical direction
        cos_theta = R[:, 2, 2]  # Third row, third column element
        # Add numerical stability
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        theta = torch.acos(cos_theta)
        return theta

    def get_true_obj(self):
        """Get objective function parameters
        Returns complete cost matrix Q diagonal and vector p
        Q: (n_state+n_ctrl)
        p: (n_state+n_ctrl,)
        """
        # Construct base diagonal weight matrix
        Q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty
        ))
        
        # Add tilt angle quadratic term
        Q[6:10] = self.tilt_Q*self.tilt_penalty
        
        # Construct p vector
        px = -torch.sqrt(self.goal_weights) * self.goal_state
        px[6:10]=-self.tilt_p*self.tilt_penalty
        p = torch.cat((px, torch.zeros(self.n_ctrl)))  # Control part is zero
        
        return Variable(Q), Variable(p)

    def get_cost_matrices(self, n_batch, mpc_T):
        """Construct batched cost matrices
        Args:
            n_batch: batch size
            mpc_T: MPC prediction horizon
        Returns:
            Q: (mpc_T, n_batch, n_state+n_ctrl, n_state+n_ctrl)
            p: (mpc_T, n_batch, n_state+n_ctrl)
        """
        # Get base q and p vectors
        q, p = self.get_true_obj()
        
        # Construct Q matrix (diagonal matrix)
        Q_diag = torch.diag(q)
        # Add tilt angle quadratic term
        Q_full = Q_diag.clone()
        Q_full[:self.n_state, :self.n_state] += self.tilt_Q
        
        # Expand to required dimensions
        Q = Q_full.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)
        p = p.unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1)
        
        return Q, p

    def get_matrices(self, X, U):
        D, D_grad_params, D_grad_x, D_grad_u=self.get_linear_dyn(X, U),self.build_batched_D_params(X,U),self.build_batched_D_x(X, U),self.build_batched_D_u(X,U)
        x_grad_theta, x_grad_xtm1, x_grad_utm1=self.build_batched_x_theta(X,U),self.build_batched_x_xtm1(X,U),self.build_batched_x_utm1(X,U)
        return D, D_grad_params, D_grad_x, D_grad_u, x_grad_theta, x_grad_xtm1, x_grad_utm1

    def grad_input(self,X, U,K=None):
        """
        Calculate gradient inputs, supporting batch processing.

        Parameters:
        X:  - time steps, batch size, state dimension
        U: - time steps, batch size, control dimension
        Returns:
        grad_D: - time steps, batch size, state dimension, control dimension + state dimension, parameter dimension
        grad_d: - time steps, batch size, state dimension, state dimension, parameter dimension
        """
        T, batch_size, n_state = X.shape
        _, _, n_ctrl = U.shape
        dx, du=n_state,n_ctrl
        dtotal=dx+du
        dtheta=len(self.params)
        # Initialize output lists grad_D and grad_d
        grad_D = []
        grad_d = []

        _X, _U = X.reshape(T * batch_size, dx), U.reshape(T * batch_size, du)
        D, D_grad_params, D_grad_x, D_grad_u, x_grad_theta, x_grad_xtm1, x_grad_utm1 = self.get_matrices(_X, _U)
        D, D_grad_params, D_grad_x, D_grad_u = D.reshape(T, batch_size, dx, dtotal), D_grad_params.reshape(T, batch_size,dx, dtotal,dtheta), D_grad_x.reshape(T, batch_size, dx, dtotal, dx), D_grad_u.reshape(T, batch_size, dx, dtotal, du)
        x_grad_theta, x_grad_xtm1, x_grad_utm1 = x_grad_theta.reshape(T, batch_size, dx, dtheta), x_grad_xtm1.reshape(T,batch_size,dx,dx), x_grad_utm1.reshape(T, batch_size, dx, du)
        # Initialize gradxt as a zero tensor
        gradxt = torch.zeros(batch_size, n_state, dtheta)
        X_U=torch.cat((X,U),dim=-1)
        d_grad_X=torch.einsum('tbnmk,tbm->tbnk',-D_grad_x,X_U)
        d_grad_U = torch.einsum('tbnmk,tbm->tbnk', -D_grad_u, X_U)

        for t in range(T):
            # Set example tensor
            if K==None:
                ut_grad_xt = torch.zeros(batch_size, n_ctrl, n_state) ####K_t
                utm1_grad_xtm1 = torch.zeros(batch_size, n_ctrl, n_state) ####K_tm1
            else:
                ut_grad_xt=K[t]
                if t>0:
                    utm1_grad_xtm1=K[t-1]

            # Calculate gradxt
            if t>0:
                gradxtm1=gradxt
                gradxt = x_grad_theta[t] + torch.matmul(x_grad_xtm1[t] + torch.matmul(x_grad_utm1[t], utm1_grad_xtm1), gradxt)

            # Calculate gradDt
            if t<(T-1):
                gradDt = D_grad_params[t] + torch.matmul(D_grad_x[t] + torch.matmul(D_grad_u[t], ut_grad_xt.unsqueeze(1)), gradxt.unsqueeze(1))
                grad_D.append(gradDt)
            if t>0:
                ####in the t+1 step, trace back the t step
                xtm1_utm1_params = torch.cat((gradxtm1, torch.matmul(utm1_grad_xtm1, gradxtm1)), dim=1)
                xtm1_utm1 = torch.cat((X[t-1], U[t-1]), dim=-1)
                grad_dtm1=gradxt-torch.einsum('bnmk,bm->bnk', grad_D[t-1],xtm1_utm1)-torch.matmul(D[t-1],xtm1_utm1_params)
                grad_d.append(grad_dtm1)

        # After the loop, convert the list to a tensor
        grad_D = torch.stack(grad_D, dim=0)
        grad_d = torch.stack(grad_d, dim=0)

        return grad_D, grad_d,D_grad_x[:T-1], D_grad_u[:T-1],D[:T-1],d_grad_X[:T-1],d_grad_U[:T-1]
    def get_linear_dyn(self, X, U):
        dt=self.dt
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())

        B = X.shape[0]
        device = X.device
        dtype = X.dtype

        # Initialize zero tensor
        D = torch.zeros((B, 13, 16), dtype=dtype, device=device)

        # Extract states
        ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        wx, wy, wz = X[:, 10], X[:, 11], X[:, 12]

        # === Fill non-zero entries (69 nnz) ===
        # Diagonal terms (13)
        for i in range(13):
            D[:, i, i] = 1.0

        # Position derivatives (3)
        D[:, 0, 3] = dt
        D[:, 1, 4] = dt
        D[:, 2, 5] = dt

        # Quaternion derivatives (columns 6-9, 24 nnz)
        # Column 6
        D[:, 3, 6] = dt * ((uz * 2 * q2 - uy * 2 * q3)) / mass
        D[:, 4, 6] = dt * ((ux * 2 * q3 - uz * 2 * q1)) / mass
        D[:, 5, 6] = dt * ((uy * 2 * q1 - ux * 2 * q2)) / mass
        D[:, 7, 6] = dt * 0.5 * wx
        D[:, 8, 6] = dt * 0.5 * wy
        D[:, 9, 6] = dt * 0.5 * wz

        # Column 7
        D[:, 3, 7] = dt * ((uy * 2 * q2 + uz * 2 * q3)) / mass
        D[:, 4, 7] = dt * ((ux * 2 * q2 - uy * 4 * q1 - uz * 2 * q0)) / mass
        D[:, 5, 7] = dt * ((ux * 2 * q3 + uy * 2 * q0 - uz * 4 * q1)) / mass
        D[:, 6, 7] = -dt * 0.5 * wx
        D[:, 8, 7] = -dt * 0.5 * wz
        D[:, 9, 7] = dt * 0.5 * wy

        # Column 8
        D[:, 3, 8] = dt * ((uy * 2 * q1 - ux * 4 * q2 + uz * 2 * q0)) / mass
        D[:, 4, 8] = dt * ((ux * 2 * q1 + uz * 2 * q3)) / mass
        D[:, 5, 8] = dt * ((uy * 2 * q3 - ux * 2 * q0 - uz * 4 * q2)) / mass
        D[:, 6, 8] = -dt * 0.5 * wy
        D[:, 7, 8] = dt * 0.5 * wz
        D[:, 9, 8] = -dt * 0.5 * wx

        # Column 9
        D[:, 3, 9] = dt * ((uz * 2 * q1 - ux * 4 * q3 - uy * 2 * q0)) / mass
        D[:, 4, 9] = dt * ((ux * 2 * q0 - uy * 4 * q3 + uz * 2 * q2)) / mass
        D[:, 5, 9] = dt * ((ux * 2 * q1 + uy * 2 * q2)) / mass
        D[:, 6, 9] = -dt * 0.5 * wz
        D[:, 7, 9] = -dt * 0.5 * wy
        D[:, 8, 9] = dt * 0.5 * wx

        # Angular velocity derivatives (columns 10-12, 18 nnz)
        # Column 10
        D[:, 6, 10] = -dt * 0.5 * q1
        D[:, 7, 10] = dt * 0.5 * q0
        D[:, 8, 10] = dt * 0.5 * q3
        D[:, 9, 10] = -dt * 0.5 * q2
        D[:, 11, 10] = -dt * ((wz * Jx - wz * Jz)) / Jy
        D[:, 12, 10] = -dt * ((wy * Jy - wy * Jx)) / Jz

        # Column 11
        D[:, 6, 11] = -dt * 0.5 * q2
        D[:, 7, 11] = -dt * 0.5 * q3
        D[:, 8, 11] = dt * 0.5 * q0
        D[:, 9, 11] = dt * 0.5 * q1
        D[:, 10, 11] = -dt * ((wz * Jz - wz * Jy)) / Jx
        D[:, 12, 11] = -dt * ((wx * Jy - wx * Jx)) / Jz

        # Column 12
        D[:, 6, 12] = -dt * 0.5 * q3
        D[:, 7, 12] = dt * 0.5 * q2
        D[:, 8, 12] = -dt * 0.5 * q1
        D[:, 9, 12] = dt * 0.5 * q0
        D[:, 10, 12] = -dt * ((wy * Jz - wy * Jy)) / Jx
        D[:, 11, 12] = -dt * ((wx * Jx - wx * Jz)) / Jy

        # Control input derivatives (columns 13-15, 11 nnz)
        # Column 13
        D[:, 3, 13] = dt * (1 - 2 * (q2 ** 2 + q3 ** 2)) / mass
        D[:, 4, 13] = dt * 2 * (q1 * q2 + q0 * q3) / mass
        D[:, 5, 13] = dt * 2 * (q1 * q3 - q0 * q2) / mass

        # Column 14
        D[:, 3, 14] = dt * 2 * (q1 * q2 - q0 * q3) / mass
        D[:, 4, 14] = dt * (1 - 2 * (q1 ** 2 + q3 ** 2)) / mass
        D[:, 5, 14] = dt * 2 * (q2 * q3 + q0 * q1) / mass
        D[:, 12, 14] = -dt * (l / 2) / Jz

        # Column 15
        D[:, 3, 15] = dt * 2 * (q1 * q3 + q0 * q2) / mass
        D[:, 4, 15] = dt * 2 * (q2 * q3 - q0 * q1) / mass
        D[:, 5, 15] = dt * (1 - 2 * (q1 ** 2 + q2 ** 2)) / mass
        D[:, 11, 15] = dt * (l / 2) / Jy

        return D

    def build_batched_x_theta(self,X, U):
        """
        Exact 1:1 conversion of CasADi's x_grad_theta to PyTorch

        Args:
            X: torch.Tensor of shape (B, 13) - state vector
            U: torch.Tensor of shape (B, 3)  - control input [ux,uy,uz]
            dt: float                        - time step
            mass: float                      - mass
            Jx, Jy, Jz: float               - moments of inertia
            l: float                         - arm length

        Returns:
            torch.Tensor of shape (B, 13, 5) - ∂x/∂θ, exactly matching CasADi's output
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt = self.dt

        # Initialize zero tensor
        x_grad_theta = torch.zeros((B, 13, 5), dtype=dtype, device=device)

        # Extract states
        ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        wx, wy, wz = X[:, 10], X[:, 11], X[:, 12]

        # === Derivative with respect to Jx (index 0) ===
        x_grad_theta[:, 10, 0] = dt * ((wy * Jz * wz - wz * Jy * wy) / (Jx * Jx))
        x_grad_theta[:, 11, 0] = -dt * (wx * wz / Jy)
        x_grad_theta[:, 12, 0] = dt * (wx * wy / Jz)

        # === Derivative with respect to Jy (index 1) ===
        x_grad_theta[:, 10, 1] = dt * (wy * wz / Jx)
        x_grad_theta[:, 11, 1] = -dt * ((l / 2 * uz - (wz * Jx * wx - wx * Jz * wz)) / (Jy * Jy))
        x_grad_theta[:, 12, 1] = -dt * (wy * wx / Jz)

        # === Derivative with respect to Jz (index 2) ===
        x_grad_theta[:, 10, 2] = -dt * (wz * wy / Jx)
        x_grad_theta[:, 11, 2] = dt * (wz * wx / Jy)
        x_grad_theta[:, 12, 2] = dt * ((l / 2 * uy + (wx * Jy * wy - wy * Jx * wx)) / (Jz * Jz))

        # === Derivative with respect to mass (index 3) ===
        mass_term = mass * mass  # Pre-compute mass squared
        x_grad_theta[:, 3, 3] = -dt * (
            ((1 - 2 * (q2 ** 2 + q3 ** 2)) * ux + 2 * (q1 * q2 - q0 * q3) * uy + 2 * (q1 * q3 + q0 * q2) * uz)
        ) / mass_term

        x_grad_theta[:, 4, 3] = -dt * (
            (2 * (q1 * q2 + q0 * q3) * ux + (1 - 2 * (q1 ** 2 + q3 ** 2)) * uy + 2 * (q2 * q3 - q0 * q1) * uz)
        ) / mass_term

        x_grad_theta[:, 5, 3] = -dt * (
            (2 * (q1 * q3 - q0 * q2) * ux + 2 * (q2 * q3 + q0 * q1) * uy + (1 - 2 * (q1 ** 2 + q2 ** 2)) * uz)
        ) / mass_term

        # === Derivative with respect to l (index 4) ===
        x_grad_theta[:, 11, 4] = dt * (0.5 * uz / Jy)
        x_grad_theta[:, 12, 4] = -dt * (0.5 * uy / Jz)

        return x_grad_theta

    def build_batched_x_utm1(self,X, U):
        """
        Exact 1:1 conversion of CasADi's x_grad_utm1 to PyTorch

        Args:
            X: torch.Tensor of shape (B, 13) - state vector [px,py,pz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            U: torch.Tensor of shape (B, 3)  - control input [ux,uy,uz]
            dt: float                        - time step
            mass: float                      - mass
            Jx, Jy, Jz: float               - moments of inertia
            l: float                         - arm length

        Returns:
            torch.Tensor of shape (B, 13, 3) - ∂x/∂u_t-1, exactly matching CasADi's output
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt = self.dt

        # Initialize zero tensor
        x_grad_utm1 = torch.zeros((B, 13, 3), dtype=dtype, device=device)

        # Extract quaternion states
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]

        # === Derivatives with respect to ux (index 0) ===
        x_grad_utm1[:, 3, 0] = dt * (1 - 2 * (q2 ** 2 + q3 ** 2)) / mass
        x_grad_utm1[:, 4, 0] = dt * 2 * (q1 * q2 + q0 * q3) / mass
        x_grad_utm1[:, 5, 0] = dt * 2 * (q1 * q3 - q0 * q2) / mass

        # === Derivatives with respect to uy (index 1) ===
        x_grad_utm1[:, 3, 1] = dt * 2 * (q1 * q2 - q0 * q3) / mass
        x_grad_utm1[:, 4, 1] = dt * (1 - 2 * (q1 ** 2 + q3 ** 2)) / mass
        x_grad_utm1[:, 5, 1] = dt * 2 * (q2 * q3 + q0 * q1) / mass
        x_grad_utm1[:, 12, 1] = -dt * (l / 2) / Jz

        # === Derivatives with respect to uz (index 2) ===
        x_grad_utm1[:, 3, 2] = dt * 2 * (q1 * q3 + q0 * q2) / mass
        x_grad_utm1[:, 4, 2] = dt * 2 * (q2 * q3 - q0 * q1) / mass
        x_grad_utm1[:, 5, 2] = dt * (1 - 2 * (q1 ** 2 + q2 ** 2)) / mass
        x_grad_utm1[:, 11, 2] = dt * (l / 2) / Jy

        return x_grad_utm1

    def build_batched_x_xtm1(self,X, U):
        """
        Exact 1:1 conversion of CasADi's x_grad_xtm1 to PyTorch

        Args:
            X: torch.Tensor of shape (B, 13) - state vector [px,py,pz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            U: torch.Tensor of shape (B, 3)  - control input [ux,uy,uz]
            dt: float                        - time step
            mass: float                      - mass
            Jx, Jy, Jz: float               - moments of inertia

        Returns:
            torch.Tensor of shape (B, 13, 13) - ∂x/∂x_t-1, exactly matching CasADi's output
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt = self.dt

        # Initialize identity matrix as base
        x_grad_xtm1 = torch.zeros((B, 13, 13), dtype=dtype, device=device)
        for i in range(13):
            x_grad_xtm1[:, i, i] = 1.0  # Diagonal elements set to 1 initially

        # Extract states
        ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        wx, wy, wz = X[:, 10], X[:, 11], X[:, 12]

        # === Position derivatives ===
        x_grad_xtm1[:, 3, 0] = dt  # ∂x[3]/∂x[0] = dt
        x_grad_xtm1[:, 4, 1] = dt  # ∂x[4]/∂x[1] = dt
        x_grad_xtm1[:, 5, 2] = dt  # ∂x[5]/∂x[2] = dt

        # === Velocity derivatives (from quaternions) ===
        # Row 6 (q0)
        x_grad_xtm1[:, 6, 3] = dt * ((uz * 2 * q2 - uy * 2 * q3)) / mass
        x_grad_xtm1[:, 6, 4] = dt * ((ux * 2 * q3 - uz * 2 * q1)) / mass
        x_grad_xtm1[:, 6, 5] = dt * ((uy * 2 * q1 - ux * 2 * q2)) / mass
        x_grad_xtm1[:, 6, 10] = dt * 0.5 * wx
        x_grad_xtm1[:, 6, 11] = dt * 0.5 * wy
        x_grad_xtm1[:, 6, 12] = dt * 0.5 * wz

        # Row 7 (q1)
        x_grad_xtm1[:, 7, 3] = dt * ((uy * 2 * q2 + uz * 2 * q3)) / mass
        x_grad_xtm1[:, 7, 4] = dt * ((ux * 2 * q2 - uy * 4 * q1 - uz * 2 * q0)) / mass
        x_grad_xtm1[:, 7, 5] = dt * ((ux * 2 * q3 + uy * 2 * q0 - uz * 4 * q1)) / mass
        x_grad_xtm1[:, 7, 10] = -dt * 0.5 * wx
        x_grad_xtm1[:, 7, 11] = -dt * 0.5 * wz
        x_grad_xtm1[:, 7, 12] = dt * 0.5 * wy

        # Row 8 (q2)
        x_grad_xtm1[:, 8, 3] = dt * ((uy * 2 * q1 - ux * 4 * q2 + uz * 2 * q0)) / mass
        x_grad_xtm1[:, 8, 4] = dt * ((ux * 2 * q1 + uz * 2 * q3)) / mass
        x_grad_xtm1[:, 8, 5] = dt * ((uy * 2 * q3 - ux * 2 * q0 - uz * 4 * q2)) / mass
        x_grad_xtm1[:, 8, 10] = -dt * 0.5 * wy
        x_grad_xtm1[:, 8, 11] = dt * 0.5 * wz
        x_grad_xtm1[:, 8, 12] = -dt * 0.5 * wx

        # Row 9 (q3)
        x_grad_xtm1[:, 9, 3] = dt * ((uz * 2 * q1 - ux * 4 * q3 - uy * 2 * q0)) / mass
        x_grad_xtm1[:, 9, 4] = dt * ((ux * 2 * q0 - uy * 4 * q3 + uz * 2 * q2)) / mass
        x_grad_xtm1[:, 9, 5] = dt * ((ux * 2 * q1 + uy * 2 * q2)) / mass
        x_grad_xtm1[:, 9, 10] = -dt * 0.5 * wz
        x_grad_xtm1[:, 9, 11] = -dt * 0.5 * wy
        x_grad_xtm1[:, 9, 12] = dt * 0.5 * wx

        # === Angular velocity derivatives ===
        # Row 10 (wx)
        x_grad_xtm1[:, 10, 6] = -dt * 0.5 * q1
        x_grad_xtm1[:, 10, 7] = dt * 0.5 * q0
        x_grad_xtm1[:, 10, 8] = dt * 0.5 * q3
        x_grad_xtm1[:, 10, 9] = -dt * 0.5 * q2
        x_grad_xtm1[:, 10, 11] = -dt * ((wz * Jx - wz * Jz)) / Jy
        x_grad_xtm1[:, 10, 12] = -dt * ((wy * Jy - wy * Jx)) / Jz

        # Row 11 (wy)
        x_grad_xtm1[:, 11, 6] = -dt * 0.5 * q2
        x_grad_xtm1[:, 11, 7] = -dt * 0.5 * q3
        x_grad_xtm1[:, 11, 8] = dt * 0.5 * q0
        x_grad_xtm1[:, 11, 9] = dt * 0.5 * q1
        x_grad_xtm1[:, 11, 10] = -dt * ((wz * Jz - wz * Jy)) / Jx
        x_grad_xtm1[:, 11, 12] = -dt * ((wx * Jy - wx * Jx)) / Jz

        # Row 12 (wz)
        x_grad_xtm1[:, 12, 6] = -dt * 0.5 * q3
        x_grad_xtm1[:, 12, 7] = dt * 0.5 * q2
        x_grad_xtm1[:, 12, 8] = -dt * 0.5 * q1
        x_grad_xtm1[:, 12, 9] = dt * 0.5 * q0
        x_grad_xtm1[:, 12, 10] = -dt * ((wy * Jz - wy * Jy)) / Jx
        x_grad_xtm1[:, 12, 11] = -dt * ((wx * Jx - wx * Jz)) / Jy

        return x_grad_xtm1

    def build_batched_D_u(self,X, U):
        """
        Build batched D_u tensor (gradient of D w.r.t. control inputs u)

        Args:
            X: torch.Tensor of shape (B, 13) - state vectors [px,py,pz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            U: torch.Tensor of shape (B, 4)  - control inputs [ux,uy,uz,tau]
            dt: float                        - time step
            mass: float                      - mass of the system
            Jx, Jy, Jz: float               - moments of inertia

        Returns:
            torch.Tensor of shape (B, 13, 16, 4) - batched D_u tensor
            where D_grad_u[b,i,j,k] = ∂D[i,j]/∂u[k] at batch b
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt = self.dt

        # Extract quaternion states
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]  # q = [q0, q1, q2, q3]

        # Initialize zero tensor
        D_grad_u = torch.zeros((B, 13, 16, 3), dtype=dtype, device=device)

        # Derivatives with respect to ux (index 0)
        D_grad_u[:, 5, 5, 0] = dt * (2 * q3 / mass)  # ∂D[5,5]/∂ux
        D_grad_u[:, 5, 6, 0] = -dt * (2 * q2 / mass)  # ∂D[5,6]/∂ux

        # Derivatives with respect to uy (index 1)
        D_grad_u[:, 6, 5, 1] = -dt * (2 * q3 / mass)  # ∂D[6,5]/∂uy
        D_grad_u[:, 6, 7, 1] = dt * (2 * q1 / mass)  # ∂D[6,7]/∂uy

        # Derivatives with respect to uz (index 2)
        D_grad_u[:, 7, 5, 2] = dt * (2 * q2 / mass)  # ∂D[7,5]/∂uz
        D_grad_u[:, 7, 6, 2] = -dt * (2 * q1 / mass)  # ∂D[7,6]/∂uz

        return D_grad_u

    def build_batched_D_x(self,X, U):
        """
        Build batched D_x tensor (gradient of D w.r.t. state variables x)

        Args:
            X: torch.Tensor of shape (B, 13) - state vectors [px,py,pz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            U: torch.Tensor of shape (B, 4) - control inputs [ux,uy,uz,tau]
            dt: float - time step
            mass: float - mass of the system
            Jx, Jy, Jz: float - moments of inertia

        Returns:
            torch.Tensor of shape (B, 13, 16, 13) - batched D_x tensor
            where D_grad_x[b,i,j,k] = ∂D[i,j]/∂x[k] at batch b
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt = self.dt

        # Initialize zero tensor
        D_grad_x = torch.zeros((B, 13, 16, 13), dtype=dtype, device=device)

        # Extract state and control variables
        ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        wx, wy, wz = X[:, 10], X[:, 11], X[:, 12]

        # Fill derivatives according to CasADi output

        # Derivatives for state variable 6 (q0) (index 5)
        D_grad_x[:, 5, 5, 6] = -dt * (2 * uz / mass) * q3
        D_grad_x[:, 5, 6, 6] = dt * 0.5
        D_grad_x[:, 5, 7, 6] = dt * 0.5
        D_grad_x[:, 5, 8, 6] = dt * 0.5

        # Derivatives for state variable 7 (q1)
        D_grad_x[:, 5, 5, 7] = dt * (2 * uy / mass) * q3
        D_grad_x[:, 6, 5, 7] = dt * (2 * ux / mass) * q2 + dt * (2 * uy / mass) * q1

        # Derivatives for state variable 8 (q2)
        D_grad_x[:, 5, 5, 8] = dt * (2 * uz / mass) * q0 - dt * (2 * ux / mass) * q3
        D_grad_x[:, 6, 5, 8] = dt * (2 * ux / mass) * q1 - dt * (4 * uy / mass) * q0 - dt * (2 * uz / mass) * q3

        # Derivatives for state variable 9 (q3)
        D_grad_x[:, 5, 5, 9] = -dt * (2 * uy / mass) * q0 + dt * (2 * ux / mass) * q1
        D_grad_x[:, 6, 5, 9] = dt * (2 * ux / mass) * q0 - dt * (2 * uz / mass) * q2

        # Derivatives for state variable 10 (wx)
        D_grad_x[:, 9, 9, 10] = -dt * 0.5
        D_grad_x[:, 10, 10, 10] = -dt * (Jy - Jx) / Jz * wy
        D_grad_x[:, 10, 11, 10] = dt * (Jx - Jz) / Jy * wz

        # Derivatives for state variable 11 (wy)
        D_grad_x[:, 9, 10, 11] = dt * 0.5
        D_grad_x[:, 11, 10, 11] = -dt * (Jy - Jx) / Jz * wx
        D_grad_x[:, 11, 12, 11] = dt * (Jz - Jy) / Jx * wz

        # Derivatives for state variable 12 (wz)
        D_grad_x[:, 9, 11, 12] = dt * 0.5
        D_grad_x[:, 9, 12, 12] = -dt * 0.5
        D_grad_x[:, 12, 11, 12] = -dt * (Jx - Jz) / Jy * wy
        D_grad_x[:, 12, 10, 12] = dt * (Jz - Jy) / Jx * wx

        return D_grad_x

    def build_batched_D_params(self,X, U):
        """
        Build batched D_params tensor (gradient of D with respect to parameters)

        Args:
            X: torch.Tensor of shape (B, 13) - state vector [px,py,pz,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            U: torch.Tensor of shape (B, 3)  - control input [ux,uy,uz]
            dt: float                        - time step
            mass: float                      - mass
            Jx, Jy, Jz: float               - moments of inertia
            l: float                         - arm length

        Returns:
            torch.Tensor of shape (B, 13, 16, 5) - batched D_params tensor
        """
        B = X.shape[0]
        D_grad_params = torch.zeros((B, 13, 16, 5), dtype=X.dtype, device=X.device)
        Jx, Jy, Jz, mass, l = torch.unbind(self.params.detach())
        dt=self.dt

        # Parameter unpacking
        q0, q1, q2, q3 = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
        wx, wy, wz = X[:, 10], X[:, 11], X[:, 12]
        ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]

        # === Derivative with respect to Jx (index 0) ===
        D_grad_params[:, 11, 10, 0] = -dt * (wz / Jy)
        D_grad_params[:, 12, 10, 0] = dt * (wy / Jz)
        D_grad_params[:, 11, 12, 0] = dt * ((wy * Jz - wy * Jy) / Jx ** 2)
        D_grad_params[:, 12, 11, 0] = dt * (wx / Jz)

        # === Derivative with respect to Jy (index 1) ===
        D_grad_params[:, 11, 10, 1] = dt * ((wz * Jx - wz * Jz) / Jy ** 2)
        D_grad_params[:, 12, 10, 1] = -dt * (wy / Jz)
        D_grad_params[:, 11, 13, 1] = dt * (wz / Jx)
        D_grad_params[:, 12, 11, 1] = -dt * (wx / Jz)
        D_grad_params[:, 11, 14, 1] = dt * (wy / Jx)
        D_grad_params[:, 12, 12, 1] = dt * ((wx * Jx - wx * Jz) / Jy ** 2)
        D_grad_params[:, 11, 15, 1] = -dt * ((l / 2) / Jy ** 2)

        # === Derivative with respect to Jz (index 2) ===
        D_grad_params[:, 11, 10, 2] = dt * (wz / Jy)
        D_grad_params[:, 12, 10, 2] = dt * ((wy * Jy - wy * Jx) / Jz ** 2)
        D_grad_params[:, 11, 13, 2] = -dt * (wz / Jx)
        D_grad_params[:, 12, 11, 2] = dt * ((wx * Jy - wx * Jx) / Jz ** 2)
        D_grad_params[:, 11, 14, 2] = -dt * (wy / Jx)
        D_grad_params[:, 12, 12, 2] = dt * (wx / Jy)
        D_grad_params[:, 12, 14, 2] = dt * ((l / 2) / Jz ** 2)

        # === Derivative with respect to mass (index 3) ===
        m2 = mass ** 2
        D_grad_params[:, 3, 13, 3] = -dt * (1 - 2 * (q2 ** 2 + q3 ** 2)) / m2
        D_grad_params[:, 4, 13, 3] = -dt * (2 * (q1 * q2 + q0 * q3)) / m2
        D_grad_params[:, 5, 13, 3] = -dt * (2 * (q1 * q3 - q0 * q2)) / m2

        D_grad_params[:, 3, 14, 3] = -dt * (2 * (q1 * q2 - q0 * q3)) / m2
        D_grad_params[:, 4, 14, 3] = -dt * (1 - 2 * (q1 ** 2 + q3 ** 2)) / m2
        D_grad_params[:, 5, 14, 3] = -dt * (2 * (q2 * q3 + q0 * q1)) / m2

        D_grad_params[:, 3, 15, 3] = -dt * (2 * (q1 * q3 + q0 * q2)) / m2
        D_grad_params[:, 4, 15, 3] = -dt * (2 * (q2 * q3 - q0 * q1)) / m2
        D_grad_params[:, 5, 15, 3] = -dt * (1 - 2 * (q1 ** 2 + q2 ** 2)) / m2

        D_grad_params[:, 3, 13, 3] += -dt * ((uz * 2 * q2 - uy * 2 * q3)) / m2
        D_grad_params[:, 4, 13, 3] += -dt * ((ux * 2 * q3 - uz * 2 * q1)) / m2
        D_grad_params[:, 5, 13, 3] += -dt * ((uy * 2 * q1 - ux * 2 * q2)) / m2

        D_grad_params[:, 3, 14, 3] += -dt * ((uy * 2 * q2 + uz * 2 * q3)) / m2
        D_grad_params[:, 4, 14, 3] += -dt * ((ux * 2 * q2 - uy * 4 * q1 - uz * 2 * q0)) / m2
        D_grad_params[:, 5, 14, 3] += -dt * ((ux * 2 * q3 + uy * 2 * q0 - uz * 4 * q1)) / m2

        D_grad_params[:, 3, 15, 3] += -dt * ((uy * 2 * q1 - ux * 4 * q2 + uz * 2 * q0)) / m2
        D_grad_params[:, 4, 15, 3] += -dt * ((ux * 2 * q1 + uz * 2 * q3)) / m2
        D_grad_params[:, 5, 15, 3] += -dt * ((uy * 2 * q3 - ux * 2 * q0 - uz * 4 * q2)) / m2

        D_grad_params[:, 3, 16 - 1, 3] += -dt * (uz * 2 * q1 - ux * 4 * q3 - uy * 2 * q0) / m2
        D_grad_params[:, 4, 16 - 1, 3] += -dt * (ux * 2 * q0 - uy * 4 * q3 + uz * 2 * q2) / m2
        D_grad_params[:, 5, 16 - 1, 3] += -dt * (ux * 2 * q1 + uy * 2 * q2) / m2

        # === Derivative with respect to l (index 4) ===
        D_grad_params[:, 11, 15, 4] = dt * 0.5 / Jy ** 2
        D_grad_params[:, 12, 14, 4] = -dt * 0.5 / Jz ** 2
        return D_grad_params

    def play_animation(self, rocket_len, state_traj, control_traj, state_traj_ref=None, control_traj_ref=None,
                       save_option=0, dt=0.1,
                       title='Rocket Powered Landing'):
        """
        Animate the rocket trajectory and control inputs in a 3D plot.

        Args:
            rocket_len: Length of the rocket body
            state_traj: State trajectory array
            control_traj: Control input trajectory array
            state_traj_ref: Reference state trajectory (optional)
            control_traj_ref: Reference control trajectory (optional)
            save_option: Whether to save the animation (0: no save, 1: save)
            dt: Time step
            title: Title of the animation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Upward (m)')
        ax.set_zlim(0, 10)
        ax.set_ylim(-8, 8)
        ax.set_xlim(-8, 8)
        ax.set_title(title, pad=10, fontsize=15)

        # Target landing point
        p = Circle((0, 0), 3, color='g', alpha=0.3)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # Data
        position = self.get_rocket_body_position(rocket_len, state_traj, control_traj)
        sim_horizon = np.size(position, 0)
        for t in range(np.size(position, 0)):
            x = position[t, 0]
            if x < 0:
                sim_horizon = t
                break
        # Animation
        line_traj, = ax.plot(position[:1, 1], position[:1, 2], position[:1, 0])
        xg, yg, zg, xh, yh, zh, xf, yf, zf = position[0, 3:]
        line_rocket, = ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')
        line_force, = ax.plot([yg, yf], [zg, zf], [xg, xf], linewidth=2, color='red')

        # Reference data
        if state_traj_ref is None or control_traj_ref is None:
            position_ref = numpy.zeros_like(position)
            sim_horizon_ref = sim_horizon
        else:
            position_ref = self.get_rocket_body_position(rocket_len, state_traj_ref, control_traj_ref)
            sim_horizon_ref = np.size((position_ref, 0))
            for t in range(np.size(position_ref, 0)):
                x = position_ref[t, 0]
                if x < 0:
                    sim_horizon_ref = t
                    break
        # Animation
        line_traj_ref, = ax.plot(position_ref[:1, 1], position_ref[:1, 2], position_ref[:1, 0], linewidth=2,
                                 color='gray', alpha=0.7)
        xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[0, 3:]
        line_rocket_ref, = ax.plot([yg_ref, yh_ref], [zg_ref, zh_ref], [xg_ref, xh_ref], linewidth=5, color='gray',
                                   alpha=0.5)
        line_force_ref, = ax.plot([yg_ref, yf_ref], [zg_ref, zf_ref], [xg_ref, xf_ref], linewidth=2, color='red',
                                  alpha=0.5)

        # Time label
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # Customize
        if state_traj_ref is not None or control_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['Reproduced', 'Demonstration'], ncol=1, loc='best',
                       bbox_to_anchor=(0.45, 0.35, 0.5, 0.5))

        def update_traj(num):
            # Customize
            time_text.set_text(time_template % (num * dt))

            # Trajectory
            if num > sim_horizon:
                t = sim_horizon
            else:
                t = num
            line_traj.set_data(position[:t, 1], position[:t, 2])
            line_traj.set_3d_properties(position[:t, 0])

            # Rocket
            xg, yg, zg, xh, yh, zh, xf, yf, zf = position[t, 3:]
            line_rocket.set_data(np.array([[yg, yh], [zg, zh]]))
            line_rocket.set_3d_properties([xg, xh])
            line_force.set_data(np.array([[yg, yf], [zg, zf]]))
            line_force.set_3d_properties([xg, xf])

            # Reference
            if num > sim_horizon_ref:
                t_ref = sim_horizon_ref
            else:
                t_ref = num
            line_traj_ref.set_data(position_ref[:t_ref, 1], position_ref[:t_ref, 2])
            line_traj_ref.set_3d_properties(position_ref[:t_ref, 0])

            # Rocket
            xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[num, 3:]
            line_rocket_ref.set_data(np.array([[yg_ref, yh_ref], [zg_ref, zh_ref]]))
            line_rocket_ref.set_3d_properties([xg_ref, xh_ref])
            line_force_ref.set_data(np.array([[yg_ref, yf_ref], [zg_ref, zf_ref]]))
            line_force_ref.set_3d_properties([xg_ref, xf_ref])

            return line_traj, line_rocket, line_force, line_traj_ref, line_rocket_ref, line_force_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, max(sim_horizon, sim_horizon_ref), interval=100, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('./videos/'+title + '.mp4', writer=writer, dpi=300)
            print('Save successful')

        plt.show()

    def get_rocket_body_position(self, rocket_len, state_traj, control_traj):
        """
        Calculate the position of the rocket body based on state and control inputs.

        Args:
            rocket_len: Length of the rocket body
            state_traj: State trajectory array
            control_traj: Control input trajectory array

        Returns:
            position: Array containing positions of rocket body points and force vectors
        """
        # Thrust position in body frame
        r_T_B = vertcat(-rocket_len / 2, 0, 0)

        # Horizon
        horizon = np.size(control_traj, 0)
        # For normalization in the plot
        norm_f = np.linalg.norm(control_traj, axis=1)
        max_f = np.amax(norm_f)
        position = np.zeros((horizon, 12))
        for t in range(horizon):
            # Position of COM
            rc = state_traj[t, 0:3]
            # Altitude of quaternion
            q = state_traj[t, 6:10]
            q = q / (np.linalg.norm(q) + 0.0001)
            # Thrust force
            f = control_traj[t, 0:3]

            # Direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # Position of gimbal point (rocket tail)
            rg = rc + mtimes(CIB, r_T_B).full().flatten()
            # Position of rocket tip
            rh = rc - mtimes(CIB, r_T_B).full().flatten()

            # Direction of force
            df = np.dot(CIB, f) / max_f
            rf = rg - df

            # Store
            position[t, 0:3] = rc
            position[t, 3:6] = rg
            position[t, 6:9] = rh
            position[t, 9:12] = rf

        return position

    def dir_cosine(self, q):
        """
        Calculate direction cosine matrix from quaternion.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            C_B_I: Direction cosine matrix from body to inertial frame
        """
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

if __name__ == '__main__':
    # Test code
    rocket = RocketDx()
    batch, T = 50, 20
    U = torch.zeros(T, batch, rocket.n_ctrl)
    X = torch.ones(T, batch, rocket.n_state)
    
    # Set initial state
    X[:,:,6] = 1.0  # Set quaternion w component to 1
    X[:,:,2] = 10  # Set initial height to 10
    X[:,0,:]=torch.tensor([10. , -8. ,  5. , -0.1,  0. , -0. ,  1. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ])
    
    # Test forward propagation
    new_state = rocket(X[0], U[0])
    print("New state shape:", new_state.shape)
    print('Hessian',rocket.tilt_Q)
    # # Test tilt angle quadratic approximation
    # print("\nTesting tilt angle quadratic approximation:")
    # theta_true, theta_approx = rocket.get_tilt_angle_quadratic_test()

    # Test dynamics
    data=np.load('Rocket_Demo.npy',allow_pickle=True)
    ctrl_traj = data.item()['demos'][0]['control_traj_opt']
    traj=data.item()['demos'][0]['state_traj_opt']
    x0=torch.as_tensor(traj[0])
    traj_our=torch.zeros(41,13)
    traj_our[0]=x0
    x=x0.float()
    for i in range (40):
        u=torch.as_tensor(ctrl_traj[i]).float()
        x=rocket.forward(x,u)
        traj_our[i+1]=x

    print(traj_our[1])
    print(traj[1])
    traj_our=traj_our.numpy()
    # rocket.play_animation(rocket_len=2, dt=0.1, state_traj=traj_our, control_traj=ctrl_traj)
    print('Error is:', np.max(np.abs(traj_our - traj)))

    # n_batch, T = 1, 20
    #
    # # Construct new xinit, including quaternion and angular velocity initialization
    # xinit = torch.zeros(n_batch, rocket.n_state, dtype=torch.float32)
    # xinit = xinit.clone()  # Avoid inplace modification of original tensor
    # xinit[:, 6:10] = torch.tensor([1.0, 0.1, 0.2, 0.1])  # Initialize as unit quaternion
    # xinit[:, 10:] = torch.tensor([0.07, 0.05, 0.04])  # Zero angular velocity
    # xinit.requires_grad = True
    #
    # uinit = torch.zeros(n_batch, rocket.n_ctrl, dtype=torch.float32)
    # uinit = uinit.clone()  # Avoid inplace
    # uinit[:, :] = torch.tensor([0.01, 0.01, 0.01])
    # uinit.requires_grad = True
    #
    # # Get CasADi analytical solution D (shape B × 13 × 16)
    # D = rocket.get_linear_dyn(xinit, uinit)
    # print("Analytical D:", D.shape)
    #
    # # Use PyTorch automatic differentiation to compute gradients (∂x_next / ∂xinit and ∂x_next / ∂uinit)
    # newx = rocket.forward(xinit, uinit)  # Output shape: (B, 13)
    #
    # Rt, St = [], []
    # for j in range(newx.shape[1]):  # Loop through each component of the next state
    #     Rj, Sj = torch.autograd.grad(
    #         newx[:, j].sum(), [xinit, uinit],
    #         retain_graph=True
    #     )
    #     Rt.append(Rj)
    #     St.append(Sj)
    #
    # Rt = torch.stack(Rt, dim=0).squeeze(1).unsqueeze(0)  # [1, 13, 13]
    # St = torch.stack(St, dim=0).squeeze(1).unsqueeze(0)  # [1, 13, 3]
    #
    # # Concatenate into PyTorch automatic differentiation version D
    # D_auto = torch.cat([Rt, St], dim=-1)  # (B, 13, 16)
    # print("Automatic differentiation D:", D_auto.shape)
    # # Compare two Ds
    # print("Difference:", torch.norm(D - D_auto).item())
    import sys

    sys.path.append('../')
    from mpc_explicit import MPC
    from mpc_explicit import GradMethods
    # #from mpc.dynamics import NNDynamics
    # #import mpc.util as eutil
    from env_dx import pendulum, cartpole
    from definitions import QuadCost
    import matplotlib
    matplotlib.use('TkAgg')
    dx=RocketDx()

    n_batch, T = 1, 40
    q, p = dx.get_true_obj()
    # q.requires_grad_()
    # p.requires_grad_()
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(T, n_batch, 1)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[0] = x0
    x = xinit
    u_init = None

    action_history=[]
    traj_history=[]
    traj_history.append(x)
    for i in range(T):
        nominal_states, nominal_actions, nominal_objs = MPC(
            dx.n_state, dx.n_ctrl, T,
            u_init=u_init,
            u_lower=None, u_upper=None,
            lqr_iter=100,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=dx.linesearch_decay,
            max_linesearch_iter=dx.max_linesearch_iter,
            grad_method=GradMethods.ANALYTIC,
            eps=1e-2,
        )(x, QuadCost(Q, p), dx)

        next_action = nominal_actions[0]
        action_history.append(next_action)
        u_init = torch.cat((nominal_actions[1:], nominal_actions[-1:]), dim=0)
        x = dx(x, next_action)
        traj_history.append(x)
        # print(q.requires_grad)
        # print(q.grad)

        # loss = (u_true - u_mpc).pow(2).mean()
        # start = time.time()
        # loss.backward()
        # end_time = time.time() - start
    action_history = torch.stack(action_history).detach()[:,0,:].numpy()
    traj_history = torch.stack(traj_history).detach()[:,0,:].numpy()
    # print(nominal_states)

    # traj_history=np.load('rocket_traj.npy')
    # action_history = np.load('rocket_ctrl.npy')
    dx.play_animation(rocket_len=2, dt=0.1, state_traj=traj_history, control_traj=action_history)


