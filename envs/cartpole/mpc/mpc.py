import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from collections import namedtuple
from enum import Enum

from . import util
from .lqr_step import LQRStep
from .dynamics import CtrlPassthroughDynamics

QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)


class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


class SlewRateCost(Module):
    """Hacky way of adding the slew rate penalty to costs."""

    # TODO: It would be cleaner to update this to just use the slew
    # rate penalty instead of # slew_C
    def __init__(self, cost, slew_C, n_state, n_ctrl):
        super().__init__()
        self.cost = cost
        self.slew_C = slew_C
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def forward(self, tau):
        true_tau = tau[:, self.n_ctrl:]
        true_cost = self.cost(true_tau)
        # The slew constraints are time-invariant.
        slew_cost = 0.5 * util.bquad(tau, self.slew_C[0])
        return true_cost + slew_cost

    def grad_input(self, x, u):
        raise NotImplementedError("Implement grad_input")


class MPC(Module):
    """A differentiable box-constrained iLQR solver.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        lqr_iter: The number of LQR iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_zero_I=None,
            u_init=None,
            lqr_iter=10,
            grad_method=GradMethods.ANALYTIC,
            delta_u=None,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            prev_ctrl=None,
            not_improved_lim=5,
            best_cost_eps=1e-4,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            pnqp_iter=20,
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.device = device

        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)

        self.u_zero_I = util.detach_maybe(u_zero_I)
        self.u_init = util.detach_maybe(u_init)
        self.lqr_iter = lqr_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps
        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl
        self.pnqp_iter = pnqp_iter

    def forward(self, x_init, cost, dx, log_iterations=False):
        # Type Checking
        if not (isinstance(cost, QuadCost) or isinstance(cost, Module) or isinstance(cost, Function)):
            raise ValueError("Invalid cost type.")
        if not (isinstance(dx, LinDx) or isinstance(dx, Module) or isinstance(dx, Function)):
            raise ValueError("Invalid dynamics type.")

        # Batch Size Inference
        n_batch = self.n_batch if self.n_batch \
            else (cost.C.size(1) if (isinstance(cost, QuadCost) and cost.C.ndimension() == 4) else cost.n_batch)
        if n_batch is None:
            raise ValueError('MPC Error: Could not infer batch size, pass in as n_batch.')

        # Process and reshape QuadCost
        if isinstance(cost, QuadCost):
            C, c = cost
            C_dims, c_dims = C.ndimension(), c.ndimension()

            # Add necessary dimensions based on the shape of C and c
            if C_dims == 2:  # Add the time and batch dimensions.
                C = C.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, self.n_state + self.n_ctrl, -1)
            elif C_dims == 3:  # Add the batch dimension.
                C = C.unsqueeze(1).expand(self.T, n_batch, self.n_state + self.n_ctrl, -1)
            if c_dims == 1:  # Add the time and batch dimensions.
                c = c.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, -1)
            elif c_dims == 2:  # Add the batch dimension.
                c = c.unsqueeze(1).expand(self.T, n_batch, -1)

            if C_dims != 4 or c_dims != 3:
                raise ValueError('MPC Error: Unexpected QuadCost shape.')

            cost = QuadCost(C, c)
        # if isinstance(cost, Module):
        #     C, c = cost.Ct, cost.ct
        #     C = C.unsqueeze(0).unsqueeze(1).expand(self.T, n_batch, self.n_state + self.n_ctrl, -1)
        #     c = c.unsqueeze(0).unsqueeze(1).expand(self.T, n_batch, -1)

        assert x_init.ndimension() == 2 and x_init.size(0) == n_batch

        # Setup initial state and control sequences
        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl, dtype=x_init.dtype, device=self.device)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x_init.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format
                  (torch.mean(util.get_cost(self.T, u, cost, dx, x_init=x_init)).item()))

        # Optimization Loop
        best = None
        if log_iterations:
            iter_log = {
                'x': [x_init],
                'u': [u],
                'cost': [util.get_cost(self.T, u, cost, dx, x_init=x_init)],
            }

        n_not_improved = 0
        for iter_num in range(self.lqr_iter):
            u = util.detach_maybe(u).clone().requires_grad_(True)

            # Compute the trajectory given the current control sequence
            x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)

            if iter_num == 0:
                costs = util.get_cost(self.T, u, cost, dx, x_init=x_init)
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                    'full_du_norm': torch.zeros(n_batch, dtype=x.dtype, device=self.device),
                }

            # Linearize the dynamics around the current trajectory & approximate cost
            F, f = (dx.F, dx.f) if isinstance(dx, LinDx) \
                else self.linearize_dynamics(x, util.detach_maybe(u), dx, diff=False)
            if isinstance(cost, QuadCost):
                C, c = cost.C, cost.c
            elif isinstance(cost, Module):
                C, c = cost.C, cost.c
            else:
                C, c, _ = self.approximate_cost(
                    x, util.detach_maybe(u), cost, diff=False)

            # Solve LQR subproblem
            x, u, n_total_qp_iter, costs, full_du_norm, mean_alphas, alpha_du_norm = \
                (self.solve_lqr_subproblem(x_init, C, c, F, f, cost, dx, x, u))
            n_not_improved += 1

            assert x.ndimension() == 3
            assert u.ndimension() == 3

            if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                    'full_du_norm': full_du_norm,
                }
            else:
                for j in range(n_batch):
                    if costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:, j].unsqueeze(1)
                        best['u'][j] = u[:, j].unsqueeze(1)
                        best['costs'][j] = costs[j]
                        best['full_du_norm'][j] = full_du_norm[j]

            if self.verbose > 0:
                print(f'iter {iter_num}: mean(cost): {torch.mean(best["costs"]).item():.2e}'
                        f' ||full_du||_max: {max(best["full_du_norm"]).item():.2e}'
                        f' ||alpha_du||_max: {max(alpha_du_norm):.2e}'
                        f' mean(alphas): {mean_alphas:.2e}'
                        f' total_qp_iters: {n_total_qp_iter}')
                util.table_log('lqr', (
                    ('iter', iter_num),
                    ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'),
                    ('||full_du||_max', max(full_du_norm).item(), '{:.2e}'),
                    ('||alpha_du||_max', max(alpha_du_norm), '{:.2e}'),
                    # TODO: alphas, total_qp_iters here is for the current iterate, not the best
                    ('mean(alphas)', mean_alphas.item(), '{:.2e}'),
                    ('total_qp_iters', n_total_qp_iter),
                ))

            if log_iterations:
                iter_log[x].append(x.detach())
                iter_log[u].append(u.detach())
                iter_log[cost].append(costs.detach())

            if max(full_du_norm) < self.eps or n_not_improved > self.not_improved_lim:
                break

        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1)
        full_du_norm = best['full_du_norm']

        # Post Optimization
        F, f = (dx.F, dx.f) if isinstance(dx, LinDx) else self.linearize_dynamics(x, u, dx, diff=True)
        if isinstance(cost, QuadCost):
            C, c = cost.C, cost.c
        elif isinstance(cost, Module):
            C, c = cost.C, cost.c
        else:
            C, c, _ = self.approximate_cost(x, u, cost, diff=True)
        x, u = self.solve_lqr_subproblem(x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)

        is_converged = full_du_norm < self.eps
        if self.detach_unconverged and max(best['full_du_norm']) > self.eps:
            if self.exit_unconverged:
                assert False
            if self.verbose >= 0:
                print("LQR Warning: All examples did not converge to a fixed point."
                      "Detaching and *not* backpropping through the bad examples.")

            I = is_converged
            Ix = I.unsqueeze(0).unsqueeze(2).expand_as(x).type_as(x.data)
            Iu = I.unsqueeze(0).unsqueeze(2).expand_as(u).type_as(u.data)
            x = x * Ix + x.clone().detach() * (1. - Ix)
            u = u * Iu + u.clone().detach() * (1. - Iu)

        costs = best['costs']

        return (x, u, costs, iter_num + 1, is_converged.detach()) if not log_iterations else \
            (x, u, costs, iter_num + 1, is_converged.detach(), iter_log)

    def solve_lqr_subproblem(self, x_init, C, c, F, f, cost, dynamics, x, u, no_op_forward=False):
        # TODO: i think "or isinstance(cost, Module)" should be removed
        if self.slew_rate_penalty is None:  # or isinstance(cost, Module):
            _lqr = LQRStep(
                n_state=self.n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=cost,
                true_dynamics=dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
                pnqp_iter=self.pnqp_iter,
            )
            e = torch.tensor([])
            return _lqr(x_init, C, c, F, f if f is not None else e)
        else:
            nsc = self.n_state + self.n_ctrl
            _n_state = nsc
            _nsc = _n_state + self.n_ctrl
            n_batch = C.size(1)
            half_gamI = (self.slew_rate_penalty *
                         torch.eye(self.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(self.T, n_batch, 1, 1))

            _C = torch.zeros(self.T, n_batch, _nsc, _nsc, dtype=C.dtype, device=self.device)
            _C[:, :, :self.n_ctrl, :self.n_ctrl] = half_gamI
            _C[:, :, -self.n_ctrl:, :self.n_ctrl] = -half_gamI
            _C[:, :, :self.n_ctrl, -self.n_ctrl:] = -half_gamI
            _C[:, :, -self.n_ctrl:, -self.n_ctrl:] = half_gamI
            slew_C = _C.clone()
            _C = _C + torch.nn.ZeroPad2d((self.n_ctrl, 0, self.n_ctrl, 0))(C)

            _c = torch.cat((torch.zeros(self.T, n_batch, self.n_ctrl, dtype=c.dtype, device=self.device), c), 2)

            _F0 = torch.cat((torch.zeros(self.n_ctrl, self.n_state + self.n_ctrl), torch.eye(self.n_ctrl),), 1).type_as(
                F).unsqueeze(0).unsqueeze(0).repeat(self.T - 1, n_batch, 1, 1)

            _F1 = torch.cat((torch.zeros(self.T - 1, n_batch, self.n_state, self.n_ctrl, dtype=F.dtype, device=self.device), F), 3)

            _F = torch.cat((_F0, _F1), 2)

            _f = torch.cat((torch.zeros(self.T - 1, n_batch, self.n_ctrl, dtype=f.dtype, device=self.device), f), 2) \
                if f is not None else torch.tensor([], device=self.device)

            u_data = util.detach_maybe(u)
            prev_u = self.prev_ctrl if self.prev_ctrl else torch.zeros(1, n_batch, self.n_ctrl, dtype=u.dtype, device=self.device)
            if prev_u.ndimension() == 1:
                prev_u = prev_u.unsqueeze(0)
            utm1s = torch.cat((prev_u, u_data[:-1]))
            _x = torch.cat((utm1s, x), 2)
            _x_init = torch.cat((prev_u[0], x_init), 1)

            _dynamics = CtrlPassthroughDynamics(dynamics) if not isinstance(dynamics, LinDx) else None

            if isinstance(cost, QuadCost):
                _true_cost = QuadCost(_C, _c)
            elif isinstance(cost, Module):
                cost.C = _C
                cost.c = _c
                _true_cost = cost
            else:
                _true_cost = SlewRateCost(cost, slew_C, self.n_state, self.n_ctrl)

            _lqr = LQRStep(
                n_state=_n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=_true_cost,
                true_dynamics=_dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=_x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
                pnqp_iter=self.pnqp_iter,
            )
            x, *rest = _lqr(_x_init, _C, _c, _F, _f)
            x = x[:, :, self.n_ctrl:]
            return [x] + rest

    def approximate_cost(self, x, u, Cf, diff=True):
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2)
            tau.requires_grad_()
            if self.slew_rate_penalty is not None:
                raise NotImplementedError(
                    "Using a non-convex cost with a slew rate penalty is not yet implemented."
                    "The current implementation does not correctly do a line search."
                    "More details: https://github.com/locuslab/mpc.pytorch/issues/12"
                )

            costs, hessians, grads = [], [], []
            for t in range(self.T):
                tau_t = tau[t]
                cost = Cf(tau_t)

                # Compute gradient
                grad = torch.autograd.grad(cost.sum(), tau_t, create_graph=True)[0]

                # Compute Hessian
                hessian = [torch.autograd.grad(grad[:, v_i].sum(), tau_t, retain_graph=True)[0]
                           for v_i in range(tau.shape[2])]
                hessian = torch.stack(hessian, dim=-1)

                # Store results
                costs.append(cost)
                grads.append(grad - util.bmv(hessian, tau_t))
                hessians.append(hessian)

            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)

            return (hessians.clone(), grads.clone(), costs.clone()) if not diff else (hessians, grads, costs)

    def linearize_dynamics(self, x, u, dynamics, diff):

        n_batch = x[0].size(0)

        if self.grad_method == GradMethods.ANALYTIC:
            _u = Variable(u[:-1].view(-1, self.n_ctrl), requires_grad=True)
            _x = Variable(x[:-1].contiguous().view(-1, self.n_state), requires_grad=True)

            # This inefficiently calls dynamics again, but is worth it because
            # we can efficiently compute grad_input for every time step at once.
            _new_x = dynamics(_x, _u)

            if not diff:
                _new_x = _new_x.clone()
                _x = _x.clone()
                _u = _u.clone()

            R, S = dynamics.grad_input(_x, _u)

            f = _new_x - util.bmv(R, _x) - util.bmv(S, _u)
            f = f.view(self.T - 1, n_batch, self.n_state)

            R = R.contiguous().view(self.T - 1, n_batch, self.n_state, self.n_state)
            S = S.contiguous().view(self.T - 1, n_batch, self.n_state, self.n_ctrl)
            F = torch.cat((R, S), 3)

            return (F.data, f.data) if not diff else (F, f)
        else:
            x = [x[0]]
            F, f = [], []
            for t in range(self.T):
                if t < self.T - 1:
                    xt = x[t].clone().requires_grad_()
                    ut = u[t].clone().requires_grad_()

                    new_x = dynamics(xt, ut)

                    # Linear dynamics approximation.
                    if self.grad_method in [GradMethods.AUTO_DIFF, GradMethods.ANALYTIC_CHECK]:
                        # Linear dynamics approximation using autograd
                        Rt, St = [], []
                        for j in range(self.n_state):
                            Rj, Sj = torch.autograd.grad(
                                new_x[:, j].sum(), [xt, ut],
                                retain_graph=True)
                            if not diff:
                                Rj, Sj = Rj.data, Sj.data
                            Rt.append(Rj)
                            St.append(Sj)
                        Rt = torch.stack(Rt, dim=1)
                        St = torch.stack(St, dim=1)
                        # Rt, St = zip(*[torch.autograd.grad(new_x[:, j].squeeze(), [xt, ut], retain_graph=True)
                        #                for j in range(self.n_state)])
                        # Rt, St = torch.stack(Rt, dim=1), torch.stack(St, dim=1)
                        # NOTE: I've replaced the latter with the following

                        # Rt, St = torch.autograd.functional.jacobian(dynamics, (xt, ut))
                        # Rt, St = Rt.squeeze(2), St.squeeze(2)

                    elif self.grad_method == GradMethods.FINITE_DIFF:
                        # Linear dynamics approximation using finite difference
                        Rt, St = zip(*[(util.jacobian(lambda s: dynamics(s, ut[i]), xt[i], 1e-4),
                                        util.jacobian(lambda a: dynamics(xt[i], a), ut[i], 1e-4))
                                       for i in range(n_batch)])
                        Rt, St = torch.stack(Rt), torch.stack(St)
                    else:
                        raise ValueError("Invalid gradient method specified.")

                    Ft = torch.cat((Rt, St), 2)
                    F.append(Ft)

                    if not diff:
                        xt, ut, new_x = xt.detach(), ut.detach(), new_x.detach()

                    ft = new_x - util.bmv(Rt, xt) - util.bmv(St, ut)
                    f.append(ft)

                if t < self.T - 1:
                    x.append(util.detach_maybe(new_x))

            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            return (F.clone(), f.clone()) if not diff else (F, f)
