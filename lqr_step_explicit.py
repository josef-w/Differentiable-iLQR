import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import scipy.linalg

from collections import namedtuple

import time

import util, mpc_backup
from pnqp import pnqp
from definitions import QuadCost, LinDx

LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)

def LQRStep(n_state,
            n_ctrl,
            T,
            u_lower=None,
            u_upper=None,
            u_zero_I=None,
            delta_u=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            true_cost=None,
            true_dynamics=None,
            delta_space=True,
            current_x=None,
            current_u=None,
            verbose=0,
            back_eps=1e-3,
            no_op_forward=False,theta=None):
    """A single step of the box-constrained iLQR solver.

        Required Args:
            n_state, n_ctrl, T
            x_init: The initial state [n_batch, n_state]

        Optional Args:
            u_lower, u_upper: The lower- and upper-bounds on the controls.
                These can either be floats or shaped as [T, n_batch, n_ctrl]
                TODO: Better support automatic expansion of these.
            TODO
        """
    # @profile
    def lqr_backward(ctx, C, c, F, f):
        n_batch = C.size(1)

        u = ctx.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(T-1, -1, -1):
            if t == T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1,2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
                        Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if u_lower is None:
                if n_ctrl == 1 and u_zero_I is None:
                    Kt = -(1./Qt_uu)*Qt_ux
                    kt = -(1./Qt_uu.squeeze(2))*qt_u
                else:
                    if u_zero_I is None:
                        Qt_uu_inv = [
                            torch.pinverse(Qt_uu[i]) for i in range(Qt_uu.shape[0])
                        ]
                        Qt_uu_inv = torch.stack(Qt_uu_inv)
                        Kt = -Qt_uu_inv.bmm(Qt_ux)
                        kt = util.bmv(-Qt_uu_inv, qt_u)

                        # Qt_uu_LU = Qt_uu.lu()
                        # Kt = -Qt_ux.lu_solve(*Qt_uu_LU)
                        # kt = -qt_u.lu_solve(*Qt_uu_LU)
                    else:
                        # Solve with zero constraints on the active controls.
                        I = u_zero_I[t].float()
                        notI = 1-I

                        qt_u_ = qt_u.clone()
                        qt_u_[I.bool()] = 0

                        Qt_uu_ = Qt_uu.clone()

                        if I.is_cuda:
                            notI_ = notI.float()
                            Qt_uu_I = (1-util.bger(notI_, notI_)).type_as(I)
                        else:
                            Qt_uu_I = 1-util.bger(notI, notI)

                        Qt_uu_[Qt_uu_I.bool()] = 0.
                        Qt_uu_[util.bdiag(I).bool()] += 1e-8

                        Qt_ux_ = Qt_ux.clone()
                        Qt_ux_[I.unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0.

                        if n_ctrl == 1:
                            Kt = -(1./Qt_uu_)*Qt_ux_
                            kt = -(1./Qt_uu.squeeze(2))*qt_u_
                        else:
                            Qt_uu_LU_ = Qt_uu_.lu()
                            Kt = -Qt_ux_.lu_solve(*Qt_uu_LU_)
                            kt = -qt_u_.unsqueeze(2).lu_solve(*Qt_uu_LU_).squeeze(2)
            else:
                assert delta_space
                lb = get_bound('lower', t) - u[t]
                ub = get_bound('upper', t) - u[t]
                if delta_u is not None:
                    lb[lb < -delta_u] = -delta_u
                    ub[ub > delta_u] = delta_u
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
                    Qt_uu, qt_u, lb, ub,
                    x_init=prev_kt, n_iter=20)
                if verbose > 1:
                    print('  + n_qp_iter: ', n_qp_iter+1)
                n_total_qp_iter += 1+n_qp_iter
                prev_kt = kt
                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[(1-If).unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0
                if n_ctrl == 1:
                    # Bad naming, Qt_uu_free_LU isn't the LU in this case.
                    Kt = -((1./Qt_uu_free_LU)*Qt_ux_)
                else:
                    Kt = -Qt_ux_.lu_solve(*Qt_uu_free_LU)

            Kt_T = Kt.transpose(1,2)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks, n_total_qp_iter


    # @profile
    def lqr_forward(ctx, x_init, C, c, F, f, Ks, ks):
        x = ctx.current_x
        u = ctx.current_u
        n_batch = C.size(1)

        old_cost = util.get_cost(T, u, true_cost, true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None

        i = 0
        while (current_cost is None or \
            (old_cost is not None and \
                torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
            i < max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(T):
                t_rev = T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)

                # Currently unimplemented:
                assert not ((delta_u is not None) and (u_lower is None))

                if u_zero_I is not None:
                    new_ut[u_zero_I[t]] = 0.

                if u_lower is not None:
                    lb = get_bound('lower', t)
                    ub = get_bound('upper', t)

                    if delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - delta_u
                        ub = u[t] + delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    # TODO(eugenevinitsky) why do we need to do this here?
                    new_ut = util.eclamp(new_ut, lb, ub)
                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < T-1:
                    if isinstance(true_dynamics, LinDx):
                        F, f = true_dynamics.F, true_dynamics.f
                        new_xtp1 = util.bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                    else:
                        new_xtp1 = true_dynamics(
                            Variable(new_xt), Variable(new_ut)).data

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                if isinstance(true_cost, QuadCost):
                    C, c = true_cost.C, true_cost.c
                    obj = 0.5*util.bquad(new_xut, C[t]) + util.bdot(new_xut, c[t])
                else:
                    obj = true_cost(new_xut)

                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u-new_u).transpose(1,2).contiguous().view(
                    n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= linesearch_decay
            i += 1

        # If the iteration limit is hit, some alphas
        # are one step too small.
        alphas[current_cost > old_cost] /= linesearch_decay
        alpha_du_norm = (u-new_u).transpose(1,2).contiguous().view(
            n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            torch.mean(alphas),
            current_cost
        )


    def get_bound(side, t):
        if side == 'lower':
            v = u_lower
        if side == 'upper':
            v = u_upper
        if isinstance(v, float):
            return v
        else:
            return v[t]

    def KKT_gradient(r,C, c, F, f, new_x, new_u,dx_init,I):
        _mpc = mpc_backup.MPC(
            n_state, n_ctrl, T,
            u_zero_I=I,
            u_init=None,
            lqr_iter=1,
            verbose=-1,
            n_batch=C.size(1),
            delta_u=None,
            # exit_unconverged=True, # It's really bad if this doesn't converge.
            exit_unconverged=False,  # It's really bad if this doesn't converge.
            eps=back_eps,
        )
        dx, du, _ = _mpc(dx_init, QuadCost(C, -r), LinDx(F,
                                                         None))  ####Obtain KKT, and use the big matrix to caculate gradient (pretend to calculate traj).

        dx, du = dx.data, du.data
        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)

        dC = torch.zeros_like(C)
        for t in range(T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
            dC[t] = dCt

        dc = -dxu

        lams = []
        prev_lam = None
        for t in range(T - 1, -1, -1):
            Ct_xx = C[t, :, :n_state, :n_state]
            Ct_xu = C[t, :, :n_state, n_state:]
            ct_x = c[t, :, :n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                lamt += util.bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))

        dlams = []
        prev_dlam = None
        for t in range(T - 1, -1, -1):
            dCt_xx = C[t, :, :n_state, :n_state]
            dCt_xu = C[t, :, :n_state, n_state:]
            drt_x = -r[t, :, :n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                dlamt += util.bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))

        dF = torch.zeros_like(F)
        for t in range(T - 1):
            xut = xu[t]
            lamt = lams[t + 1]

            dxut = dxu[t]
            dlamt = dlams[t + 1]

            dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))

        if f.nelement() > 0:
            _dlams = dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            df = torch.Tensor()

        dx_init = -dlams[0]
        return dx_init, dC, dc, dF, df

    def KKT_gradient_cuda(r, C, c, F, f, new_x, new_u, dx_init, I):
        # Move inputs to GPU
        r = r.cuda()
        C = C.cuda()
        c = c.cuda()
        F = F.cuda()
        f = f.cuda()
        new_x = new_x.cuda()
        new_u = new_u.cuda()
        dx_init = dx_init.cuda()
        if I is not None:
            I = I.cuda()

        # Move MPC model to GPU
        _mpc = mpc_backup.MPC(
            n_state, n_ctrl, T,
            u_zero_I=I,
            u_init=None,
            lqr_iter=1,
            verbose=-1,
            n_batch=C.size(1),
            delta_u=None,
            exit_unconverged=False,  # It's really bad if this doesn't converge.
            eps=back_eps,
        ).cuda()  # Transfer the MPC model to GPU

        # Run the MPC and results already on GPU
        dx, du, _ = _mpc(dx_init, QuadCost(C, -r), LinDx(F, None))
        dx, du = dx.data, du.data

        # Concatenate dx and du without additional .cuda() call
        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)

        # Initialize dC on GPU
        dC = torch.zeros_like(C)
        for t in range(T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
            dC[t] = dCt

        dc = -dxu

        # Compute the Lagrange multipliers
        lams = []
        prev_lam = None
        for t in range(T - 1, -1, -1):
            Ct_xx = C[t, :, :n_state, :n_state]
            Ct_xu = C[t, :, :n_state, n_state:]
            ct_x = c[t, :, :n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                lamt += util.bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))

        # Compute the gradient of the Lagrange multipliers
        dlams = []
        prev_dlam = None
        for t in range(T - 1, -1, -1):
            dCt_xx = C[t, :, :n_state, :n_state]
            dCt_xu = C[t, :, :n_state, n_state:]
            drt_x = -r[t, :, :n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                dlamt += util.bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))

        # Initialize dF on GPU
        dF = torch.zeros_like(F)
        for t in range(T - 1):
            xut = xu[t]
            lamt = lams[t + 1]

            dxut = dxu[t]
            dlamt = dlams[t + 1]

            dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))

        if f.nelement() > 0:
            _dlams = dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            df = torch.Tensor()

        dx_init = -dlams[0]

        # Move final results back to CPU
        return dx_init.cpu(), dC.cpu(), dc.cpu(), dF.cpu(), df.cpu()

    def fix_point_equ(dl_dx, dl_du, matrices, theta_shape):
        # print(tau_grad_D)
        T, batch_size, dx, du, dtheta = dl_dx.shape[0], dl_dx.shape[1], dl_dx.shape[2], dl_du.shape[2], theta_shape
        d_total = dx + du
        new_batch_size = batch_size * T * (d_total)
        dtau_dC, dtau_dc, dtau_dF, dtau_df, grad_D, grad_d, D_grad_X, D_grad_U, D, d_grad_X, d_grad_U = matrices
        # print(dtau_dC.requires_grad)
        # dtau_dF = torch.ones(T - 1, new_batch_size, dx, d_total)
        # dtau_df = torch.ones(T - 1, new_batch_size, dx)
        tau_grad_D, tau_grad_d = dtau_dF, dtau_df
        D_grad_theta, d_grad_theta = grad_D, grad_d

        # D_grad_X = torch.ones(T - 1, batch_size, dx, d_total, dx)
        # D_grad_U = torch.randn(T - 1, batch_size, dx, d_total, du)
        # D = torch.randn((T - 1), batch_size, dx, (dx + du))
        # D_grad_theta = torch.randn(T - 1, batch_size, dx, d_total, dtheta)*1.3
        # d_grad_theta = torch.randn(T - 1, batch_size, dx, dtheta)
        # tau_grad_D, tau_grad_d=torch.randn(dtau_dF.shape),torch.randn(dtau_df.shape)
        ######sorting the deminsions
        X_grad_D = tau_grad_D.view(T - 1, batch_size, T, d_total, dx, d_total).permute(1, 2, 3, 0, 4, 5)[:, :, :dx, :,
                   :, :].reshape(batch_size, T * dx, T - 1, dx, d_total)
        U_grad_D = tau_grad_D.view(T - 1, batch_size, T, d_total, dx, d_total).permute(1, 2, 3, 0, 4, 5)[:, :, dx:, :,
                   :, :].reshape(batch_size, T * du, T - 1, dx, d_total)

        X_grad_d = tau_grad_d.view(T - 1, batch_size, T, d_total, dx).permute(1, 2, 3, 0, 4)[:, :, :dx, :, :].reshape(
            batch_size, T * dx, T - 1, dx)
        U_grad_d = tau_grad_d.view(T - 1, batch_size, T, d_total, dx).permute(1, 2, 3, 0, 4)[:, :, dx:, :, :].reshape(
            batch_size, T * du, T - 1, dx)

        X_grad_C = dtau_dC.reshape(T, batch_size, T, d_total, d_total, d_total).permute(1, 2, 3, 0, 4, 5)[:, :, :dx, :,
                   :, :].reshape(batch_size, T * dx, T * d_total * d_total)
        U_grad_C = dtau_dC.reshape(T, batch_size, T, d_total, d_total, d_total).permute(1, 2, 3, 0, 4, 5)[:, :, dx:, :,
                   :, :].reshape(batch_size, T * du, T * d_total * d_total)

        X_grad_c = dtau_dc.view(T, batch_size, T, d_total, d_total).permute(1, 2, 3, 0, 4)[:, :, :dx, :, :].reshape(
            batch_size, T * dx, T * d_total)
        U_grad_c = dtau_dc.view(T, batch_size, T, d_total, d_total).permute(1, 2, 3, 0, 4)[:, :, dx:, :, :].reshape(
            batch_size, T * du, T * d_total)
        #######obtain dd_dx and dd_du
        ########obtain terms for the equation
        D_grad_X = D_grad_X.permute(1, 0, 2, 3, 4)
        D_grad_U = D_grad_U.permute(1, 0, 2, 3, 4)
        d_grad_X = d_grad_X.permute(1, 0, 2, 3)
        d_grad_U = d_grad_U.permute(1, 0, 2, 3)
        D_grad_theta = D_grad_theta.permute(1, 0, 2, 3, 4)
        d_grad_theta = d_grad_theta.permute(1, 0, 2, 3)

        X_D_X = torch.einsum('bptnm,btnmk->bptk', X_grad_D, D_grad_X)
        U_D_U = torch.einsum('bptnm,btnmk->bptk', U_grad_D, D_grad_U)
        X_d_X = torch.einsum('bptn,btnk->bptk', X_grad_d, d_grad_X)
        U_d_U = torch.einsum('bptn,btnk->bptk', U_grad_d, d_grad_U)

        X_D_U = torch.einsum('bptnm,btnmk->bptk', X_grad_D, D_grad_U)
        U_D_X = torch.einsum('bptnm,btnmk->bptk', U_grad_D, D_grad_X)
        X_d_U = torch.einsum('bptn,btnk->bptk', X_grad_d, d_grad_U)
        U_d_X = torch.einsum('bptn,btnk->bptk', U_grad_d, d_grad_X)

        X_D_X = torch.cat((X_D_X, torch.zeros(batch_size, dx * T, 1, dx)), dim=2)
        U_D_U = torch.cat((U_D_U, torch.zeros(batch_size, du * T, 1, du)), dim=2)
        X_D_U = torch.cat((X_D_U, torch.zeros(batch_size, dx * T, 1, du)), dim=2)
        U_D_X = torch.cat((U_D_X, torch.zeros(batch_size, du * T, 1, dx)), dim=2)

        X_d_X = torch.cat((X_d_X, torch.zeros(batch_size, dx * T, 1, dx)), dim=2)
        U_d_U = torch.cat((U_d_U, torch.zeros(batch_size, du * T, 1, du)), dim=2)
        X_d_U = torch.cat((X_d_U, torch.zeros(batch_size, dx * T, 1, du)), dim=2)
        U_d_X = torch.cat((U_d_X, torch.zeros(batch_size, du * T, 1, dx)), dim=2)

        X_grad_X = (X_D_X + X_d_X).reshape(batch_size, dx * T, dx * T)
        U_grad_U = (U_D_U + U_d_U).reshape(batch_size, du * T, du * T)
        X_grad_U = (X_D_U + X_d_U).reshape(batch_size, dx * T, du * T)
        U_grad_X = (U_D_X + U_d_X).reshape(batch_size, du * T, dx * T)

        # X_grad_X = (X_D_X+ X_d_X).reshape(batch_size, dx * T, dx * T)
        # U_grad_U = (U_D_U+ U_d_U).reshape(batch_size, du * T, du * T)
        # X_grad_U = (X_D_U).reshape(batch_size, dx * T, du * T)
        # U_grad_X = (U_D_X).reshape(batch_size, du * T, dx * T)
        # print(X_d_X.reshape(batch_size, dx * T, dx * T)[0])
        # print(X_D_X.reshape(batch_size, dx * T, dx * T)[0])

        X_D_theta = torch.einsum('bptnm,btnmk->bpk', X_grad_D, D_grad_theta)
        X_d_theta = torch.einsum('bptn,btnk->bpk', X_grad_d, d_grad_theta)
        U_D_theta = torch.einsum('bptnm,btnmk->bpk', U_grad_D, D_grad_theta)
        U_d_theta = torch.einsum('bptn,btnk->bpk', U_grad_d, d_grad_theta)
        F_theta = X_D_theta + X_d_theta
        G_theta = U_D_theta + U_d_theta
        #########solving linear equation
        I_X = torch.eye(dx * T, dx * T).unsqueeze(0).repeat(batch_size, 1, 1)
        I_U = torch.eye(du * T, du * T).unsqueeze(0).repeat(batch_size, 1, 1)

        A_X = I_X - X_grad_X
        B_U = I_U - U_grad_U
        # rank1 = torch.linalg.matrix_rank(A_X[0])
        # print(f'matrix rank: {rank1}')
        A = torch.zeros(batch_size, T * d_total, T * d_total)  # 创建一个足够大的零矩阵以容纳所有项
        B = torch.zeros(batch_size, T * d_total, dtheta)  # 创建对应的 B 矩阵

        A[:, :T * dx, :T * dx] = A_X
        A[:, :T * dx, T * dx:] = -X_grad_U
        A[:, T * dx:, :T * dx] = -U_grad_X
        A[:, T * dx:, T * dx:] = B_U

        lambda_ = 1e-5
        A_reg = A + lambda_ * torch.eye(T * d_total, T * d_total).unsqueeze(0).repeat(batch_size, 1, 1)
        rank = torch.linalg.matrix_rank(A[1])
        # print(A)
        print(f'matrix rank: {rank}')
        B[:, :T * dx, :] = F_theta
        B[:, T * dx:, :] = G_theta
        # print(G_theta)
        # print(A)
        # solution, residuals, rank, singular_values = torch.linalg.lstsq(A_reg, B)
        ####solve dtheta
        solution = torch.linalg.solve(A, B)
        X_theta = solution[:, :T * dx, :]
        U_theta = solution[:, T * dx:, :]
        ####solve dC,dc
        ##solve dC
        B = torch.zeros(batch_size, T * d_total, T * d_total * d_total)
        B[:, :T * dx, :] = X_grad_C
        B[:, T * dx:, :] = U_grad_C
        solution = torch.linalg.lstsq(A, B)[0]
        X_C = solution[:, :T * dx, :]
        U_C = solution[:, T * dx:, :]
        ##solve dc
        B = torch.zeros(batch_size, T * d_total, T * d_total)
        B[:, :T * dx, :] = X_grad_c
        B[:, T * dx:, :] = U_grad_c
        solution = torch.linalg.lstsq(A, B)[0]
        X_c = solution[:, :T * dx, :]
        U_c = solution[:, T * dx:, :]
        #######connect with the loss function
        dl_dx = dl_dx.permute(1, 0, 2).reshape(batch_size, T * dx)
        dl_du = dl_du.permute(1, 0, 2).reshape(batch_size, T * du)
        dtheta = torch.einsum('bi,bij->bj', dl_dx, X_theta) + torch.einsum('bi,bij->bj', dl_du, U_theta)

        dC = torch.einsum('bi,bij->bj', dl_dx, X_C) + torch.einsum('bi,bij->bj', dl_du, U_C)
        dc = torch.einsum('bi,bij->bj', dl_dx, X_c) + torch.einsum('bi,bij->bj', dl_du, U_c)
        dC, dc = dC.reshape(batch_size, T, d_total, d_total).permute(1, 0, 2, 3), dc.reshape(batch_size, T,
                                                                                             d_total).permute(1, 0, 2)
        # dC, dc=None,None
        return dC, dc, dtheta

    class LQRStepFn(Function):
        # @profile
        @staticmethod
        def forward(ctx, x_init, C, c, F, f=None,theta=None,if_converge=False):
            if no_op_forward:
                assert current_x is not None
                assert current_u is not None
                c_back = []
                for t in range(T):
                    xt = current_x[t]
                    ut = current_u[t]
                    xut = torch.cat((xt, ut), 1)
                    c_back.append(util.bmv(C[t], xut) + c[t])
                c_back = torch.stack(c_back)
                f_back = None
                ctx.current_x = current_x
                ctx.current_u = current_u
                Ks, ks, n_total_qp_iter = lqr_backward(ctx, C, c_back, F, f_back)
                Ks=torch.stack(Ks,dim=0)
                ctx.save_for_backward(
                    x_init, C, c, F, f, current_x, current_u,theta,Ks)
                ctx.current_x, ctx.current_u = current_x, current_u

                return current_x, current_u

            if delta_space:
                # Taylor-expand the objective to do the backward pass in
                # the delta space.
                assert current_x is not None
                assert current_u is not None
                c_back = []
                for t in range(T):
                    xt = current_x[t]
                    ut = current_u[t]
                    xut = torch.cat((xt, ut), 1)
                    c_back.append(util.bmv(C[t], xut) + c[t])
                c_back = torch.stack(c_back)
                f_back = None
            else:
                assert False

            ctx.current_x = current_x
            ctx.current_u = current_u

            Ks, ks, n_total_qp_iter = lqr_backward(ctx, C, c_back, F, f_back)
            new_x, new_u, for_out = lqr_forward(ctx,
                x_init, C, c, F, f, Ks, ks)
            ctx.save_for_backward(x_init, C, c, F, f, new_x, new_u)

            return new_x, new_u, torch.Tensor([n_total_qp_iter]), \
              for_out.costs, for_out.full_du_norm, for_out.mean_alphas

        @staticmethod
        def backward(ctx, dl_dx, dl_du, temp=None, temp2=None):
            T, batch_size, dx, du = dl_dx.shape[0],dl_dx.shape[1],dl_dx.shape[2],dl_du.shape[2]
            start = time.time()
            x_init, C, c, F, f, new_x, new_u,theta,Ks = ctx.saved_tensors
            theta_shape=theta.shape[0]
            r = []
            for t in range(T):
                rt = torch.cat((dl_dx[t], dl_du[t]), 1)
                r.append(rt)
            r = torch.stack(r)

            d_total = dx + du
            new_batch_size = batch_size * T * (d_total)
            r_new = torch.zeros(T, new_batch_size, d_total)
            for t in range(T):
                for b in range(batch_size):
                    for d in range(d_total):
                        index = b * T * (d_total) + t * (d_total) + d
                        r_new[t, index, d] = 1

            # 复制张量
            C_new = C.detach().repeat_interleave(T * d_total, dim=1)
            c_new = c.detach().repeat_interleave(T * d_total, dim=1)
            x_init_new = x_init.detach().repeat_interleave(T * d_total, dim=0)
            F_new=F.detach().repeat_interleave((T) * d_total, dim=1)
            f_new = f.detach().repeat_interleave((T) * d_total, dim=1)
            new_x_new=new_x.detach().repeat_interleave(T * d_total, dim=1)
            new_u_new = new_u.detach().repeat_interleave(T * d_total, dim=1)

            if u_lower is None:
                I = None
                I_new=None
            else:
                u_lower_new=torch.tensor(u_lower).unsqueeze(0)
                u_upper_new=torch.tensor(u_upper).unsqueeze(0)
                # u_lower_new = u_lower_new.repeat_interleave(T * d_total, dim=1)
                # u_upper_new = u_upper_new.detach().repeat_interleave(T * d_total, dim=1)
                I = (torch.abs(new_u - u_lower) <= 1e-8) | \
                    (torch.abs(new_u - u_upper) <= 1e-8)
                I_new=(torch.abs(new_u_new - u_lower_new) <= 1e-8) | \
                    (torch.abs(new_u_new - u_upper_new) <= 1e-8)

            dx_init = Variable(torch.zeros_like(x_init))
            dx_init_new = Variable(torch.zeros_like(x_init_new))

            # dx_init, dC, dc, dF, df=KKT_gradient(r,C, c, F, f, new_x, new_u, dx_init, I)
            if T*batch_size>1500:
                _,dtau_dC,dtau_dc,dtau_dF,dtau_df=KKT_gradient_cuda(r_new,C_new, c_new, F_new, f_new, new_x_new, new_u_new, dx_init_new, I_new)
            else:
                _, dtau_dC, dtau_dc, dtau_dF, dtau_df = KKT_gradient(r_new, C_new, c_new, F_new, f_new, new_x_new,new_u_new, dx_init_new, I_new)
            grad_F, grad_f, F_grad_x, F_grad_u, F, f_grad_X, f_grad_U = true_dynamics.grad_input(new_x, new_u, Ks)
            matrices=[dtau_dC,dtau_dc,dtau_dF,dtau_df,grad_F, grad_f, F_grad_x, F_grad_u, F, f_grad_X, f_grad_U]
            dC_fix,dc_fix,dtheta=fix_point_equ(dl_dx, dl_du,matrices,theta_shape)
            backward_time = time.time() - start
            print(backward_time)
            #print(dC,dc,dF,df)
            # return dx_init, dC, dc, dF, df, None,None
            # print((dC-dC_fix).sum())
            return None, dC_fix, dc_fix, None, None, dtheta, None
            # if if_converge==torch.tensor(1.):
            #     return None, None,None,None,None,dtheta,None
            # else:
            #     return dx_init, dC, dc, dF, df,None

    return LQRStepFn.apply