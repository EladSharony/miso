import os
import functools
import numpy as np
import torch

from dataclasses import dataclass

from utils import combine_logs_and_delete_temp_files, save, repeat_and_perturb, InitPredictor


@dataclass
class BOXDDPIterate:
    input_trajectory: torch.Tensor
    state_trajectory: torch.Tensor


@dataclass
class BOXDDPSolution:
    state_trajectory: torch.Tensor
    input_trajectory: torch.Tensor
    tracking_cost: torch.Tensor
    num_iterations: torch.Tensor


def select_decorator(func):
    def wrapper(*args, **kwargs):
        # Get current experiment type
        experiment = os.getenv('EXP_TYPE')
        # Check the experiment type and apply the corresponding decorator
        if experiment == 'open_loop':
            decorated_func = init_methods_open_loop_decorator(func)
        elif experiment == 'closed_loop':
            decorated_func = init_methods_closed_loop_decorator(func)
        else:
            raise ValueError(f"Invalid experiment type: {experiment}")

        # Call the decorated function
        return decorated_func(*args, **kwargs)

    return wrapper


def init_methods_closed_loop_decorator(cls):
    def wrap_method(method):
        @functools.wraps(method)
        def wrapped_method(self, current_state, goal_state):
            if not self.init_args_saved_flag:
                self.init_args_saved_flag = True
                save(metadata=self.metadata, data={key: self.metadata[key] for key in ['args', 'kwargs']})

            warm_start_iterate = self.warm_start_prev_sol()

            solution = solve(self=self, current_state=current_state,
                             goal_state=goal_state, warm_start_iterate=warm_start_iterate)

            if isinstance(solution, list):
                # reconstruct solution with batch dimension
                input_trajectory = torch.stack([sol.input_trajectory for sol in solution], dim=1)
                state_trajectory = torch.stack([sol.state_trajectory for sol in solution], dim=1)
                num_iterations = torch.stack([sol.num_iterations for sol in solution], dim=0)
                tracking_cost = torch.stack([sol.tracking_cost for sol in solution], dim=0)
            else:
                input_trajectory = solution.input_trajectory
                state_trajectory = solution.state_trajectory
                num_iterations = solution.num_iterations
                tracking_cost = solution.tracking_cost

            self.previous_solution = BOXDDPIterate(input_trajectory=input_trajectory,
                                                   state_trajectory=state_trajectory)

            solution = BOXDDPSolution(input_trajectory=input_trajectory,
                                      state_trajectory=state_trajectory,
                                      tracking_cost=tracking_cost,
                                      num_iterations=num_iterations)

            self.metadata['iteration'] += 1

            return solution

        return wrapped_method

    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.method = self.metadata['method']
            self.metadata['iteration'] = 0
            # log args and kwargs
            self.metadata['args'] = args
            self.metadata['kwargs'] = kwargs
            self.init_args_saved_flag = self.method != 'warm_start'
            if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
                self.predict = InitPredictor(self.metadata)

            # Wrap specified methods.yaml
            for method_name in ['solve']:
                if hasattr(self, method_name):
                    original_method = getattr(self, method_name)
                    wrapped = wrap_method(original_method)
                    setattr(self, method_name, wrapped.__get__(self, type(self)))

    return WrappedClass


def init_methods_open_loop_decorator(cls):
    def wrap_method(method):
        @functools.wraps(method)
        def wrapped_method(self):
            if self.metadata['iteration'] >= self.metadata['iterations']:
                combine_logs_and_delete_temp_files(self.metadata)
                return False
            current_state = self.boxddp_log['current_state'][self.metadata['iteration']]
            goal_state = self.boxddp_log['goal_state'][[self.metadata['iteration']]]

            warm_start_iterate = BOXDDPIterate(
                input_trajectory=self.boxddp_log['warm_start_input_trajectory'][self.metadata['iteration']],
                state_trajectory=self.boxddp_log['warm_start_state_trajectory'][self.metadata['iteration']])

            solution = solve(self=self, current_state=current_state,
                             goal_state=goal_state, warm_start_iterate=warm_start_iterate)

            self.metadata['iteration'] += 1

            return True

        return wrapped_method

    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            cfg = args[0]
            super().__init__(cfg=cfg)
            self.method = self.metadata['method']
            self.metadata['iteration'] = 0
            load_filename = f"{cfg['load_path']}{cfg['load_name']}_{cfg['scenario_token']}"
            # solver_init = np.load(f"{load_filename}_solver_init.npz", allow_pickle=True)
            self.boxddp_log = dict(np.load(f"{load_filename}.npz", allow_pickle=True))
            # Convert the numpy arrays to torch tensors
            for key in self.boxddp_log.keys():
                self.boxddp_log[key] = torch.from_numpy(self.boxddp_log[key]).to(cfg['device'])
            self.metadata['iterations'] = self.boxddp_log['tracking_cost'].shape[0]

            if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
                self.predict = InitPredictor(self.metadata)

            # Wrap specified methods.yaml
            for method_name in ['solve']:
                if hasattr(self, method_name):
                    original_method = getattr(self, method_name)
                    wrapped = wrap_method(original_method)
                    setattr(self, method_name, wrapped.__get__(self, type(self)))

    return WrappedClass


def solve(self, current_state, goal_state, warm_start_iterate):
    if self.method in ['warm_start', 'oracle']:
        solution = run_boxddp(self=self, current_state=current_state, current_iterate=warm_start_iterate)
        # reconstruct solution with batch dimension
        num_iterations = solution.num_iterations
        tracking_cost = solution.tracking_cost
        input_trajectory = solution.input_trajectory
        state_trajectory = solution.state_trajectory

    elif self.method in ['NN', 'NN_perturbation', 'NN_ensemble', 'warm_start_perturbation']:

        if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
            current_state[:, 2] = principal_value(current_state[:, 2])
            if goal_state.ndim == 3:
                goal_state[:, :, 2] = principal_value(goal_state[:, :, 2])
            else:
                goal_state[:, 2] = principal_value(goal_state[:, 2])
            warm_start_iterate.state_trajectory[:, :, 2] = principal_value(warm_start_iterate.state_trajectory[:, :, 2])

            input_trajectory = warm_start_iterate.input_trajectory.permute(1, 0, 2)
            state_trajectory = warm_start_iterate.state_trajectory.permute(1, 0, 2)
            if goal_state.ndim == 2:
                goal_state = goal_state.unsqueeze(0)
            goal_state = goal_state.permute(1, 0, 2)

            predicted_input_trajectory = []
            for input_traj, state_traj, goal in zip(input_trajectory, state_trajectory, goal_state):
                warm_start_iter = BOXDDPIterate(input_trajectory=input_traj,
                                                state_trajectory=state_traj)
                predicted_input_trajectory.append(self.predict(reference_trajectory=goal,
                                                               warm_start_iterate=warm_start_iter))
            predicted_input_trajectory = torch.stack(predicted_input_trajectory, dim=1)
            if self.method == 'NN_perturbation':
                predicted_input_trajectories = []
                for predicted_input_traj in predicted_input_trajectory.swapaxes(0, 1):
                    predicted_input_trajectories.append(repeat_and_perturb(
                        input_trajectory=predicted_input_traj,
                        K=self.metadata['num_perturbations'],
                        noise_scale=self.metadata['noise_scale']))
                predicted_input_trajectory = torch.stack(predicted_input_trajectories, dim=1)

        elif self.method == 'warm_start_perturbation':
            predicted_input_trajectories = []
            for input_trajectory in warm_start_iterate.input_trajectory.swapaxes(0, 1):
                predicted_input_trajectories.append(repeat_and_perturb(
                    input_trajectory=input_trajectory,
                    K=self.metadata['num_perturbations'],
                    noise_scale=self.metadata['noise_scale']))
            predicted_input_trajectory = torch.stack(predicted_input_trajectories, dim=1)

        lowest_cost = torch.tensor(float('inf')).repeat(self.n_batch)
        best_solution = [[] for _ in range(self.n_batch)]
        best_iterate = [None for _ in range(self.n_batch)]
        for input_trajectory in predicted_input_trajectory.swapaxes(0, 2):  # iterate over K
            if self.metadata['optimizer_mode'] == 'multiple':
                current_iterate = BOXDDPIterate(input_trajectory=input_trajectory.permute(1, 0, 2),
                                                state_trajectory=torch.empty_like(warm_start_iterate.state_trajectory))
                solution = run_boxddp(self=self, current_state=current_state.clone(), current_iterate=current_iterate)
                for i in range(self.n_batch):
                    if solution.tracking_cost[i] < lowest_cost[i]:
                        lowest_cost[i] = solution.tracking_cost[i]
                        best_solution[i] = BOXDDPSolution(input_trajectory=solution.input_trajectory[:, i, :],
                                                          state_trajectory=solution.state_trajectory[:, i, :],
                                                          tracking_cost=solution.tracking_cost[i],
                                                          num_iterations=torch.zeros(1))

            elif self.metadata['optimizer_mode'] == 'single':
                if current_state.ndim == 1:
                    current_state = current_state.unsqueeze(0)

                state_trajectory = [current_state.clone()]
                for t in range(input_trajectory.shape[1]):
                    state_trajectory.append(
                        self.dx(state_trajectory[t], input_trajectory[:, t, :]))
                state_trajectory = torch.stack(state_trajectory, dim=1)

                xut = torch.cat((state_trajectory,
                                 torch.cat((input_trajectory, torch.zeros(input_trajectory.shape[0], 1, self.n_ctrl)), dim=1)), dim=-1)
                tracking_cost = self.cost(xut.swapaxes(0, 1).to(torch.float32)).sum(dim=0)
                for i in range(self.n_batch):
                    if tracking_cost[i] < lowest_cost[i]:
                        lowest_cost[i] = tracking_cost[i]
                        best_iterate[i] = BOXDDPIterate(input_trajectory=input_trajectory[i],
                                                        state_trajectory=state_trajectory[i])

        if self.metadata['optimizer_mode'] == 'single':
            current_iterate = BOXDDPIterate(input_trajectory=torch.stack([sol.input_trajectory for sol in best_iterate], dim=1),
                                            state_trajectory=torch.stack([sol.state_trajectory for sol in best_iterate], dim=1))
            solution = run_boxddp(self=self, current_state=current_state, current_iterate=current_iterate)
            best_solution = BOXDDPSolution(input_trajectory=solution.input_trajectory,
                                           state_trajectory=solution.state_trajectory,
                                           tracking_cost=solution.tracking_cost,
                                           num_iterations=torch.zeros(1))

        solution = best_solution

        if self.metadata['optimizer_mode'] == 'multiple':
            # reconstruct solution with batch dimension
            num_iterations = torch.stack([sol.num_iterations for sol in solution], dim=0)
            tracking_cost = torch.stack([sol.tracking_cost for sol in solution], dim=0)
            input_trajectory = torch.stack([sol.input_trajectory for sol in solution], dim=1)
            state_trajectory = torch.stack([sol.state_trajectory for sol in solution], dim=1)
        elif self.metadata['optimizer_mode'] == 'single':
            num_iterations = solution.num_iterations
            tracking_cost = solution.tracking_cost
            input_trajectory = solution.input_trajectory
            state_trajectory = solution.state_trajectory

    save(metadata=self.metadata,
         data={'current_state': current_state.numpy(force=True),
               'goal_state': self.goal_state.numpy(force=True),
               'num_iterations': np.array(num_iterations),
               'tracking_cost': tracking_cost.numpy(force=True),
               'input_trajectory': input_trajectory.numpy(force=True),
               'state_trajectory': state_trajectory.numpy(force=True),
               'warm_start_input_trajectory': warm_start_iterate.input_trajectory.numpy(force=True),
               'warm_start_state_trajectory': warm_start_iterate.state_trajectory.numpy(force=True)})

    return solution


def run_boxddp(self, current_state, current_iterate):
    if current_iterate is None:
        current_iterate = self.warm_start_prev_sol()

    # We need to pad the control input with an additional zero - due to the way mpc.pytorch is implemented
    self.mpc.u_init = torch.cat((current_iterate.input_trajectory, torch.zeros((1, self.n_batch, self.n_ctrl),
                                 device=current_iterate.input_trajectory.device)), dim=0)
    state_trajectory, input_trajectory, tracking_cost, num_iterations, is_converged = \
        self.mpc(current_state, self.cost, self.dx, log_iterations=False)
    # Remove the last element (always zero)
    input_trajectory = input_trajectory[:-1, :, :]

    return BOXDDPSolution(state_trajectory=state_trajectory, input_trajectory=input_trajectory,
                          tracking_cost=tracking_cost, num_iterations=num_iterations)


def generate_combination(seed):
    """
    Generates a reproducible unique combination of values based on a given seed.

    The generated combination consists of two lists:
    1. The first list contains four elements:
       - The first element is chosen randomly within the range [-2, 2].
       - The second element is chosen randomly within the range [-1, 1].
       - The third element is chosen randomly within the range [-π/2, π/2].
       - The fourth element is chosen randomly within the range [-π/4, π/4].
    2. The second list contains one element, chosen randomly within the range [-2, 2].

    All values are rounded to two decimal places.

    Args:
        seed (int): The seed for the random number generator to ensure reproducibility.

    Returns:
        list: A combination consisting of two lists as described above.

    Example:
        [[-0.5, 0.1, 1.57, -0.79], [1.46]]
    """
    # Set the seed for the random number generator
    np.random.seed(seed)

    def random_value(low, high, size=None):
        return np.round(np.random.uniform(low, high, size), 2).astype(np.float32)

    # First list with the specified ranges
    first_list = [
        random_value(-2, 2),          # First element in [-2, 2]
        random_value(-1, 1),          # Second element in [-1, 1]
        random_value(-np.pi/2, np.pi/2),  # Third element in [-π/2, π/2]
        random_value(-np.pi/4, np.pi/4)   # Fourth element in [-π/4, π/4]
    ]

    # Second list with one element in the range [-2, 2]
    second_list = [random_value(-2, 2)]

    combination = [first_list, second_list]
    return combination


def principal_value(angle):
    """
    Wrap heading angle between -pi and pi.
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi
