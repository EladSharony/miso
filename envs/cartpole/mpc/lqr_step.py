import torch
from torch.autograd import Function
from collections import namedtuple
from . import util, mpc
from .pnqp import pnqp

LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
LqrForOut = namedtuple('lqrForOut', 'objs full_du_norm alpha_du_norm mean_alphas costs')


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
            no_op_forward=False,
            pnqp_iter=20):
    """
    A single step of the box-constrained iLQR solver.

    Args:
    - n_state (int): Dimension of the state space.
    - n_ctrl (int): Dimension of the control space.
    - T (int): Number of time steps.

    - u_lower (float or list of tensors): Lower bound on the control inputs.
    - u_upper (float or list of tensors): Upper bound on the control inputs.
    - u_zero_I (list of tensors): List of boolean tensors indicating which controls are zero.
    - delta_u (float): Maximum change in control inputs.

    - linesearch_decay (float): Decay rate for the line search.
    - max_linesearch_iter (int): Maximum number of line search iterations.

    - true_cost (function): Function that computes the cost of a state and
        control input.
    - true_dynamics (function): Function that computes the next state given the current state and control input.

    - delta_space (bool): Whether to do the backward pass in the delta space.

    - current_x (list of tensors): List of current states.
    - current_u (list of tensors): List of current control inputs.

    - verbose (int, optional): Verbosity level. Defaults to 0.
    - back_eps (float, optional): Backward pass convergence criterion.
    - no_op_forward (bool, optional): Flag to indicate if forward pass should be skipped.

    Returns:
    - LQRStepFn (function): Function that performs a single step of the box-constrained iLQR solver.
    """

    def lqr_backward(ctx, C, c, F, f):
        """The backward pass of the LQR step.

        This function computes the feedback and feedforward gains using the linearized dynamics and the cost-to-go function.

        Args:
            ctx (autograd.Context): The context for storing information needed for the backward pass.
            C (Tensor): Quadratic cost matrices for each time step.
            c (Tensor): Linear cost vectors for each time step.
            F (Tensor): Linearized dynamics matrices for each time step.
            f (Tensor): Linearized dynamics bias terms for each time step.

        Returns:
            tuple:
                K (Tensor): Feedback gains for each time step.
                k (Tensor): Feedforward gains for each time step.
                n_total_qp_iter (int): Total number of Quadratic Programming (QP) iterations.
        """

        u = ctx.current_u    # Retrieve the current control input from the context
        Ks, ks = [], []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None

        # Backward Riccati recursion: Iterate over the time steps in reverse order
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                # If this is the last time step, the Q-function is simply the immediate cost
                Qt, qt = C[t], c[t]
            else:
                # Otherwise, compute the Q-function based on the Bellman equation using
                # the linearized dynamics and the value function derivatives for the next time step
                Ft = F[t]
                Ft_T = Ft.transpose(1, 2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

            # Partition the Q-function into its submatrices for state and control variables
            Qt_xx = Qt[:, :n_state, :n_state]  # Q-function's state-state submatrix
            Qt_xu = Qt[:, :n_state, n_state:]  # Q-function's state-control submatrix
            Qt_ux = Qt[:, n_state:, :n_state]  # Q-function's control-state submatrix
            Qt_uu = Qt[:, n_state:, n_state:]  # Q-function's control-control submatrix
            qt_x = qt[:, :n_state]  # q-function's state subvector
            qt_u = qt[:, n_state:]  # q-function's control subvector

            # Check if bounds for control are provided
            if u_lower is None:
                if n_ctrl == 1 and u_zero_I is None:
                    # Case for single control input without zero constraints
                    Kt = -(1. / Qt_uu) * Qt_ux
                    kt = -(1. / Qt_uu.squeeze(2)) * qt_u
                else:
                    if u_zero_I is None:
                        # Case for multiple control inputs without zero constraints
                        Qt_uu_inv = torch.linalg.pinv(Qt_uu)
                        Kt = -Qt_uu_inv.bmm(Qt_ux)
                        kt = util.bmv(-Qt_uu_inv, qt_u)
                    else:
                        # Solve with zero constraints on the active controls.
                        I = u_zero_I[t].bool()
                        notI = ~I

                        qt_u_ = qt_u.clone()
                        qt_u_[I.bool()] = 0

                        Qt_uu_ = Qt_uu.clone()
                        Qt_uu_I = ~util.bger(notI, notI)

                        Qt_uu_[Qt_uu_I] = 0.
                        Qt_uu_[util.bdiag(I)] += 1e-8  # Add small value to diagonal for numerical stability

                        Qt_ux_ = Qt_ux.clone()
                        Qt_ux_[I.unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0.

                        if n_ctrl == 1:
                            Kt = -(1. / Qt_uu_) * Qt_ux_
                            kt = -(1. / Qt_uu.squeeze(2)) * qt_u_
                        else:
                            Qt_uu_LU_ = torch.linalg.lu_factor(Qt_uu_)
                            Kt = -torch.linalg.lu_solve(*Qt_uu_LU_, Qt_ux_)
                            kt = -torch.linalg.lu_solve(*Qt_uu_LU_, qt_u_.unsqueeze(2)).squeeze(2)
            else:
                assert delta_space
                lb = get_bound('lower', t) - u[t]
                ub = get_bound('upper', t) - u[t]
                if delta_u is not None:
                    lb[lb < -delta_u] = -delta_u
                    ub[ub > delta_u] = delta_u

                # Projected Newton Quadratic Programming to solve for constraints
                kt, Qt_uu_constrained, If, n_qp_iter = pnqp(Qt_uu, qt_u, lb, ub, x_init=prev_kt, n_iter=pnqp_iter)

                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[~If.bool().unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0

                Kt = -((1. / Qt_uu_constrained) * Qt_ux_) if n_ctrl == 1 \
                    else -torch.linalg.lu_solve(*Qt_uu_constrained, Qt_ux_)

                if verbose > 1:
                    print('  + n_qp_iter: ', n_qp_iter + 1)
                n_total_qp_iter += 1 + n_qp_iter
                prev_kt = kt

            Kt_T = Kt.transpose(1, 2)

            Ks.append(Kt)
            ks.append(kt)

            # Calculate the value function for the next step
            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                   Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                   Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks, n_total_qp_iter

    def lqr_forward(ctx, x_init, C, c, F, f, Ks, ks):
        """
        Forward pass of the LQR algorithm.

        Args:
        - ctx: A context object containing information about the current state and control.
        - x_init: Initial state.
        - C, c: Cost matrices.
        - F, f: Dynamics matrices.
        - Ks, ks: Control matrices and control vectors.

        Returns:
        - new_x: Updated state trajectory.
        - new_u: Updated control trajectory.
        - LqrForOut: A named tuple containing various outputs.
        """
        x = ctx.current_x
        u = ctx.current_u
        n_batch = C.size(1)

        # Get the cost of the current state and control
        old_cost = util.get_cost(T, u, true_cost, true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch, dtype=C.dtype, device=C.device)
        full_du_norm = None

        i = 0
        # Iterate until the cost decreases for all batches or the maximum number of iterations is reached
        while ((current_cost is None or (old_cost is not None and (current_cost > old_cost).any().item()))
               and i < max_linesearch_iter):

            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init, dtype=x_init.dtype, device=x_init.device)]
            objs = []

            # Iterate over the time horizon
            for t in range(T):
                t_rev = T - 1 - t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + alphas.unsqueeze(1) * kt

                # Constraint handling
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
                    new_ut = torch.clamp(new_ut, lb, ub)

                new_u.append(new_ut)
                new_xut = torch.cat((new_xt, new_ut), dim=1)
                # Dynamics update
                if t < T - 1:
                    if isinstance(true_dynamics, mpc.LinDx):
                        F, f = true_dynamics.F, true_dynamics.f
                        new_xtp1 = util.bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                    else:
                        new_xtp1 = true_dynamics(new_xt, new_ut).detach()

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t + 1])

                # Cost computation
                if isinstance(true_cost, mpc.QuadCost):
                    C, c = true_cost.C, true_cost.c
                    obj = 0.5 * util.bquad(new_xut, C[t]) + util.bdot(new_xut, c[t])
                else:
                    obj = true_cost(new_xut)

                objs.append(obj)

            objs = torch.stack(objs)
            # Calculate the cumulative cost
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)

            # Calculate the norm of the control difference for the entire horizon (serves as a convergence measure)
            if full_du_norm is None:
                full_du_norm = (u - new_u).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)

            # If the newly computed cost is greater than the old cost for some batches,
            # reduce the step size (alpha) for those batches to ensure convergence
            alphas[current_cost > old_cost] *= linesearch_decay

            i += 1

        # Adjust the alphas that were reduced in the last iteration
        alphas[current_cost > old_cost] /= linesearch_decay
        # Compute the norm of the control difference scaled by the final alphas
        # (provides a measure of the control updates with respect to the step size)
        alpha_du_norm = (u - new_u).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(objs, full_du_norm, alpha_du_norm, torch.mean(alphas), current_cost)

    def get_bound(side, t):
        v = u_lower if side == 'lower' else u_upper if side == 'upper' else None
        return v if isinstance(v, float) else v[t]

    class LQRStepFn(Function):
        """
        Custom PyTorch autograd Function that implements a single step of the LQR algorithm.
        This function defines both the forward and backward passes for the LQR step.
        """
        @staticmethod
        def forward(ctx, x_init, C, c, F, f=None):
            """
             Forward pass of the LQR step.

             Args:
             - ctx: A context object for storing intermediate results needed for the backward pass.
             - x_init: Initial state.
             - C, c: Cost matrices.
             - F, f: Dynamics matrices.

             Returns:
             - new_x: Updated state trajectory.
             - new_u: Updated control trajectory.
             - Other relevant data about the optimization process.
             """

            # Check if no operation is to be performed for the forward pass
            if no_op_forward:
                # Save all inputs and current state and control for backward pass
                ctx.save_for_backward(x_init, C, c, F, f, current_x, current_u)
                ctx.current_x, ctx.current_u = current_x, current_u
                return current_x, current_u

            if delta_space:
                # If in delta space, Taylor-expand the objective for the backward pass
                assert current_x is not None
                assert current_u is not None

                # Combine current state and control for first time step
                xut = torch.cat((current_x[0], current_u[0]), 1)
                c_back0 = util.bmv(C[0], xut) + c[0]
                c_back = torch.zeros(T, *c_back0.shape, dtype=c_back0.dtype, device=c_back0.device)
                c_back[0, ...] = c_back0

                # Iterate over the time horizon to compute the expanded objective
                for t in range(T - 1):
                    xut = torch.cat((current_x[t + 1], current_u[t + 1]), 1)
                    c_back[t + 1, ...] = util.bmv(C[t + 1], xut) + c[t + 1]
                f_back = None
            else:
                assert False # Unknown space (not delta space)

            ctx.current_x = current_x
            ctx.current_u = current_u

            # Call the backward LQR function to get control matrices and vectors
            Ks, ks, n_total_qp_iter = lqr_backward(ctx, C, c_back, F, f_back)
            # Call the forward LQR function to get updated state and control trajectories
            new_x, new_u, for_out = lqr_forward(ctx, x_init, C, c, F, f, Ks, ks)
            # Save inputs and results for the backward pass
            ctx.save_for_backward(x_init, C, c, F, f, new_x, new_u)

            return new_x, new_u, torch.tensor(
                [n_total_qp_iter]), for_out.costs, for_out.full_du_norm, for_out.mean_alphas, for_out.alpha_du_norm

        @staticmethod
        def backward(ctx, dl_dx, dl_du):
            """
            Backward pass of the LQR step.

            This function computes the gradients with respect to the input parameters
            based on the gradients of the output.

            Args:
            - ctx: A context object containing saved tensors and other information from the forward pass.
            - dl_dx: Gradient of the loss w.r.t. the state.
            - dl_du: Gradient of the loss w.r.t. the control.

            Returns:
            - Gradients of the input parameters.
            """

            # Retrieve saved tensors from the forward pass
            x_init, C, c, F, f, new_x, new_u = ctx.saved_tensors

            # Initialize the residuals
            r0 = torch.cat((dl_dx[0], dl_du[0]), 1)
            r = torch.zeros(T, *r0.shape, dtype=r0.dtype, device=r0.device)
            r[0, ...] = r0
            for t in range(T - 1):
                rt = torch.cat((dl_dx[t + 1], dl_du[t + 1]), 1)
                r[t + 1, ...] = rt

            # Compute constraints if there are any
            I = None if u_lower is None else (torch.abs(new_u - u_lower) <= 1e-8) | (torch.abs(new_u - u_upper) <= 1e-8)

            # Set up the MPC controller with relevant constraints and parameters
            dx_init = torch.zeros_like(x_init)
            _mpc = mpc.MPC(n_state, n_ctrl, T,
                           u_zero_I=I, u_init=None, lqr_iter=1,
                           verbose=-1, n_batch=C.size(1),
                           delta_u=None, eps=back_eps,
                           exit_unconverged=False   # A failure to converge here would be problematic
                           )

            # Use the MPC to compute the backward dynamics
            dx, du, _ = _mpc(dx_init, mpc.QuadCost(C, -r), mpc.LinDx(F, None))
            dxu = torch.cat((dx.detach(), du.detach()), 2)
            xu = torch.cat((new_x, new_u), 2)

            # Calculate gradients with respect to cost matrices
            dC = torch.zeros_like(C)
            for t in range(T):
                xut = torch.cat((new_x[t], new_u[t]), 1)
                dxut = dxu[t]
                dCt = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
                dC[t] = dCt
            dc = -dxu

            # Compute the costate (lambda) for the entire trajectory.
            # (lambda represents the gradient of the Hamiltonian with respect to the state.)

            # Start with the final timestep since the costate for the final timestep is
            # derived directly from the cost function. There's no previous lambda (prev_lam)
            # for the last timestep, which is why it's handled separately.
            t = T - 1
            Ct_xx = C[t, :, :n_state, :n_state]
            Ct_xu = C[t, :, :n_state, n_state:]
            ct_x = c[t, :, :n_state]
            lamt = util.bmv(Ct_xx, new_x[t]) + util.bmv(Ct_xu, new_u[t]) + ct_x

            # Initialize the lambda trajectory tensor
            lams = torch.zeros(T, *lamt.shape, dtype=lamt.dtype, device=lamt.device)
            lams[t, ...] = lamt
            prev_lam = lamt

            # For the rest of the timesteps, we use the recursive relationship
            # involving the dynamics to compute the lambda trajectory backwards in time
            for t in range(T - 2, -1, -1):
                Ct_xx = C[t, :, :n_state, :n_state]
                Ct_xu = C[t, :, :n_state, n_state:]
                ct_x = c[t, :, :n_state]
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                lamt = util.bmv(Ct_xx, new_x[t]) + util.bmv(Ct_xu, new_u[t]) + ct_x + util.bmv(Fxt, prev_lam)
                lams[t, ...] = lamt
                prev_lam = lamt

            # Compute the gradient of lambda (dlamt) through the trajectory.
            # This involves the gradient of the Hamiltonian w.r.t. both state and control.

            # Handle the final timestep separately for the same reason as before
            t = T - 1
            dCt_xx = C[t, :, :n_state, :n_state]
            dCt_xu = C[t, :, :n_state, n_state:]
            drt_x = -r[t, :, :n_state]
            dlamt = util.bmv(dCt_xx, dx[t]) + util.bmv(dCt_xu, du[t]) + drt_x

            # Initialize the gradient of lambda trajectory tensor
            dlams = torch.zeros(T, *dlamt.shape, dtype=dlamt.dtype, device=dlamt.device)
            dlams[t, ...] = dlamt
            prev_dlam = dlamt

            # Compute the gradient of lambda trajectory for the remaining timesteps
            for t in range(T - 2, -1, -1):
                dCt_xx = C[t, :, :n_state, :n_state]
                dCt_xu = C[t, :, :n_state, n_state:]
                drt_x = -r[t, :, :n_state]
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                dlamt = util.bmv(dCt_xx, dx[t]) + util.bmv(dCt_xu, du[t]) + drt_x + util.bmv(Fxt, prev_dlam)
                dlams[t, ...] = dlamt
                prev_dlam = dlamt

            # Compute the gradient with respect to the dynamics matrices
            dF = - (util.bger(dlams[1:, ...].squeeze(1), xu[:-1, ...].squeeze(1)) +
                    util.bger(lams[1:, ...].squeeze(1), dxu[:-1, ...].squeeze(1))).unsqueeze(1)

            # Compute the gradient with respect to the dynamics bias if it exists
            if f.nelement() > 0:
                _dlams = dlams[1:]
                assert _dlams.shape == f.shape
                df = -_dlams
            else:
                df = torch.Tensor()

            dx_init = -dlams[0]

            return dx_init, dC, dc, dF, df

    return LQRStepFn.apply
