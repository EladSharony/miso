import numpy as np
import torch

if __name__ == '__main__':
    from mpc.mpc import MPC, GradMethods
    from boxddp_utils import (select_decorator, BOXDDPSolution, BOXDDPIterate, combine_logs_and_delete_temp_files,
                              generate_combination)
else:
    from .mpc.mpc import MPC, GradMethods
    from .boxddp_utils import (select_decorator, BOXDDPSolution, BOXDDPIterate, combine_logs_and_delete_temp_files, generate_combination)


class EnvDx(torch.nn.Module):
    def __init__(self, device: str, n_substeps=2, n_batch=16):
        super().__init__()
        self.device = torch.device(device)

        self.nx = 4
        self.nu = 1
        self.n_batch = n_batch
        self.n_substeps = n_substeps
        self.dt = 0.1

        self.u_lower = -5.5
        self.u_upper = 5.5

        self.dtype = torch.double

        self.f = self.dynamics(self.dt, self.u_lower, self.u_upper)

    @staticmethod
    def dynamics(dt, u_lower=-10.0, u_upper=10.0):
        m_c = 1.0
        m_p = 0.3
        l = 0.5
        g = -9.81

        def f(t, y, args):
            # gym
            k = (t / dt - 1).clamp(min=0).floor().int()
            u = args[0][range(args[0].size(0)), k, ...].squeeze(-1)
            u = torch.clamp(u, u_lower, u_upper)

            x, x_dot, theta, theta_dot = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)

            temp = (u + l * m_p * theta_dot ** 2 * sin_theta) / (m_c + m_p)
            thetaacc = (-g * sin_theta - cos_theta * temp) / (l * (4.0 / 3.0 - m_p * cos_theta ** 2 / (m_c + m_p)))
            xacc = temp - l * m_p * thetaacc * cos_theta / (m_c + m_p)

            x = x + dt * x_dot
            x_dot = x_dot + dt * xacc

            theta = theta + dt * theta_dot
            theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

            theta_dot = theta_dot + dt * thetaacc

            return torch.stack((x, x_dot, theta, theta_dot), dim=-1)

        return f

    def forward(self, state, ctrl):
        state, ctrl = state.to(self.dtype), ctrl.to(self.dtype)
        if ctrl.ndim == 3:
            self.n_batch, n_steps, _ = ctrl.shape
        else:
            n_steps = 1
            self.n_batch, _ = ctrl.shape

        if n_steps == 1:
            next_state = [state]
            for t in range(self.n_substeps):
                next_state.append(self.f(torch.tensor(0), next_state[-1], (ctrl,)))
            return next_state[-1].to(torch.float32)

        else:
            raise NotImplementedError("Unrolling the entire trajectory is not implemented")

    def state_diff(self, state1, state2):
        """
        Compute the difference between two states, taking into account the periodicity
        """
        diff = state1 - state2
        diff[..., 2] = self.principal_value(diff[..., 2])
        return diff

    @staticmethod
    def principal_value(angle):
        """
        Wrap angle between -pi and pi.
        """
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi


class Cost(torch.nn.Module):
    def __init__(self, dx, goal_state, goal_weights, ctrl_weights, T, n_batch=1):
        super().__init__()
        self.dx = dx
        self.device = self.dx.device
        self.n_batch = n_batch

        self.T = T
        self.n_state = self.dx.nx
        self.n_ctrl = self.dx.nu
        self.goal_weights = goal_weights
        self.goal_state = goal_state

        # Define the quadratic cost matrices Q and R
        self.Q = torch.diag(goal_weights).to(self.device)
        self.R = ctrl_weights * torch.eye(self.n_ctrl, device=self.device)

        # Define the linear cost terms
        self.q = -goal_weights * goal_state
        self.r = torch.zeros((self.n_batch, self.n_ctrl), device=self.device)

        # Combine Q and R into Ct and ct as before
        self.Ct = torch.cat((torch.cat((self.Q, torch.zeros(self.n_state, self.n_ctrl, device=self.device)), dim=1),
                             torch.cat((torch.zeros(self.n_ctrl, self.n_state, device=self.device),
                                        self.R), dim=1)), dim=0)
        self.ct = torch.cat((self.q, self.r), dim=1)

        # Expand the cost terms for the entire horizon and batch size
        self.C = self.Ct[None, ...].expand(T, n_batch, -1, -1)
        self.c = self.ct[None, ...].expand(T, n_batch, -1)

    def forward(self, xut):
        # Split state and control
        x = xut[..., :self.n_state]
        u = xut[..., self.n_state:]

        # Compute the quadratic cost
        x_quad = 0.5 * torch.sum((x - self.goal_state) @ self.Q * (x - self.goal_state), dim=-1)
        u_quad = 0.5 * torch.sum(u @ self.R * u, dim=-1)

        # Compute the linear cost
        # x_lin = torch.sum(self.q * (x - self.goal_state), dim=-1)
        # u_lin = torch.sum(self.r * u, dim=-1)

        # Total cost
        cost = x_quad + u_quad  #+ x_lin + u_lin
        return cost

    def update_goal(self):
        self.ct = torch.cat((-self.goal_weights * self.goal_state, torch.zeros(self.n_ctrl, device=self.device)), dim=0)
        self.C = self.Ct[None, ...].expand(self.T, self.n_batch, -1, -1)
        self.c = self.ct[None, ...].expand(self.T, self.n_batch, -1)


@select_decorator
class BoxDDPSolver:
    def __init__(self, cfg):
        self.metadata = cfg
        self.device = cfg['device']
        self.seed = cfg['seed']

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_device(self.device)

        initial_states, goal_states = [], []
        for _ in range(cfg['mpc']['n_batch']):
            initial_state, goal_state = generate_combination(cfg['scenario_token'] * cfg['mpc']['n_batch'] + _)
            initial_states.append(initial_state)
            goal_states.append(goal_state)

        cfg['env']['initial_state'], cfg['env']['goal_state'] = torch.tensor(initial_states), torch.tensor([
                                                                                                               goal_state + [
                                                                                                                   0, 0,
                                                                                                                   0]
                                                                                                               for
                                                                                                               goal_state
                                                                                                               in
                                                                                                               goal_states])

        self.dx = EnvDx(**{k: v for k, v in cfg['env'].items() if k not in ['T_env', 'goal_state', 'initial_state']})

        self.n_state = self.dx.nx
        self.n_ctrl = self.dx.nu
        self.n_batch = self.dx.n_batch

        self.goal_state = cfg['env']['goal_state']
        goal_weights = torch.tensor(cfg['mpc']['goal_weights'])
        ctrl_weights = torch.tensor(cfg['mpc']['ctrl_weights'])

        self.cost = Cost(self.dx, self.goal_state, goal_weights, ctrl_weights, cfg['mpc']['T'], cfg['mpc']['n_batch'])

        cfg[
            'mpc'].update({'n_state': self.n_state, 'n_ctrl': self.n_ctrl, 'u_lower': self.dx.u_lower,
                           'u_upper': self.dx.u_upper, 'grad_method': GradMethods(cfg['mpc']['grad_method'])})
        self.mpc = MPC(**{k: v for k, v in cfg['mpc'].items() if k not in ['goal_weights', 'ctrl_weights']})

        self.previous_solution = None

    def solve(self, current_state, goal_state=None, current_iterate=None):
        if current_iterate is None:
            current_iterate = self.warm_start_prev_sol()

        # We need to pad the control input with an additional zero - due to the way mpc.pytorch is implemented
        self.mpc.u_init = torch.cat((current_iterate.input_trajectory, torch.zeros(self.n_batch, self.n_ctrl)), dim=0)
        state_trajectory, input_trajectory, tracking_cost, num_iterations, is_converged = self.mpc(current_state, self.cost, self.dx, log_iterations=False)
        # Remove the last element (always zero)
        input_trajectory = input_trajectory[:-1, :, :]

        solution = BOXDDPSolution(state_trajectory=state_trajectory, input_trajectory=input_trajectory, tracking_cost=tracking_cost, num_iterations=num_iterations)
        self.previous_solution = solution

        return solution

    def warm_start_prev_sol(self):
        # if the previous solution is None, return a zero input trajectory
        if self.previous_solution is None:
            return BOXDDPIterate(input_trajectory=torch.zeros(self.mpc.T - 1, self.n_batch, self.n_ctrl), state_trajectory=torch.zeros(self.mpc.T, self.n_batch, self.n_state))

        # Otherwise, shift and pad with zeros
        input_trajectory = torch.cat((self.previous_solution.input_trajectory[1:, :, :],
                                      torch.zeros(1, self.n_batch, self.n_ctrl)))
        state_trajectory = torch.cat((self.previous_solution.state_trajectory[1:, :, :],
                                      torch.zeros(1, self.n_batch, self.n_state)))

        return BOXDDPIterate(input_trajectory=input_trajectory, state_trajectory=state_trajectory)


class Env:
    def __init__(self, cfg, scenario_token):
        self.T_env = cfg['env']['T_env']
        cfg['scenario_token'] = scenario_token
        self.solver = BoxDDPSolver(cfg)
        self.initial_state = cfg['env']['initial_state']

    def run_simulation(self):
        state = [self.initial_state]
        for _ in range(self.T_env):
            solution = self.solver.solve(current_state=state[-1], goal_state=self.solver.goal_state)
            # Advance the state by one step
            state.append(solution.state_trajectory[1, ...])
        combine_logs_and_delete_temp_files(self.solver.metadata)
