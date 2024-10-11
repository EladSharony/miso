import os
import functools

import numpy as np
import torch

from dataclasses import dataclass
from utils import combine_logs_and_delete_temp_files, save, repeat_and_perturb, InitPredictor


@dataclass
class MPPIIterate:
    input_trajectory: torch.Tensor
    state_trajectory: torch.Tensor


@dataclass
class MPPISolution:
    state_trajectory: torch.Tensor
    input_trajectory: torch.Tensor
    tracking_cost: torch.Tensor


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

            self.previous_solution = MPPIIterate(input_trajectory=solution.input_trajectory,
                                                 state_trajectory=solution.state_trajectory)

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
            current_state = self.mppi_log['current_state'][self.metadata['iteration']]
            goal_state = self.mppi_log['goal_state'][self.metadata['iteration']]

            warm_start_iterate = MPPIIterate(
                input_trajectory=self.mppi_log['warm_start_input_trajectory'][self.metadata['iteration']],
                state_trajectory=self.mppi_log['warm_start_state_trajectory'][self.metadata['iteration']])

            solution = solve(self=self, current_state=current_state,
                             goal_state=goal_state, warm_start_iterate=warm_start_iterate)

            self.metadata['iteration'] += 1

            return True

        return wrapped_method

    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            cfg = args[0]
            load_filename = f"{cfg['load_path']}{cfg['load_name']}_{cfg['scenario_token']}"
            solver_init = np.load(f"{load_filename}_solver_init.npz", allow_pickle=True)
            env = solver_init['args'][0]
            cfg['goal_state'] = solver_init['args'][1]['goal_state']
            super().__init__(env=env, cfg=cfg)
            self.method = self.metadata['method']
            self.metadata['iteration'] = 0
            self.mppi_log = dict(np.load(f"{load_filename}.npz", allow_pickle=True))
            # Convert the numpy arrays to torch tensors
            for key in self.mppi_log.keys():
                self.mppi_log[key] = torch.from_numpy(self.mppi_log[key])
            self.metadata['iterations'] = self.mppi_log['tracking_cost'].shape[0]

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
        solution = run_mppi(self=self, current_state=current_state, current_iterate=warm_start_iterate)

    elif self.method in ['NN', 'NN_perturbation', 'NN_ensemble', 'warm_start_perturbation']:
        if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
            predicted_input_trajectory = self.predict(reference_trajectory=goal_state,
                                                      warm_start_iterate=warm_start_iterate)

            if self.method == 'NN_perturbation':
                predicted_input_trajectory = repeat_and_perturb(
                    input_trajectory=predicted_input_trajectory,
                    K=self.metadata['num_perturbations'],
                    noise_scale=self.metadata['noise_scale'])

        elif self.method == 'warm_start_perturbation':
            predicted_input_trajectory = repeat_and_perturb(
                input_trajectory=warm_start_iterate.input_trajectory,
                K=self.metadata['num_perturbations'],
                noise_scale=self.metadata['noise_scale'])

        lowest_cost = float('inf')
        for input_trajectory in predicted_input_trajectory.swapaxes(0, 1):  # iterate over K
            if self.metadata['optimizer_mode'] == 'multiple':
                current_iterate = MPPIIterate(input_trajectory=input_trajectory.clone(),
                                              state_trajectory=torch.empty_like(warm_start_iterate.state_trajectory))
                solution = run_mppi(self=self, current_state=current_state.clone(),
                                    current_iterate=current_iterate)
                if solution.tracking_cost < lowest_cost:
                    lowest_cost = solution.tracking_cost
                    best_solution = MPPISolution(input_trajectory=solution.input_trajectory,
                                                 state_trajectory=solution.state_trajectory,
                                                 tracking_cost=solution.tracking_cost)

            elif self.metadata['optimizer_mode'] == 'single':
                state_trajectory = [current_state.clone().unsqueeze(0)]
                for t in range(input_trajectory.shape[0]):
                    state_trajectory.append(self.dx(state_trajectory[t], input_trajectory.clone()[[t], :]))
                state_trajectory = torch.cat(state_trajectory, dim=-2)

                tracking_cost = self.mpc.running_cost(state_trajectory, None).sum()
                if tracking_cost < lowest_cost:
                    lowest_cost = tracking_cost
                    best_iterate = MPPIIterate(input_trajectory=input_trajectory.clone(),
                                               state_trajectory=torch.empty_like(warm_start_iterate.state_trajectory))

        if self.metadata['optimizer_mode'] == 'single':
            solution = run_mppi(self=self, current_state=current_state.clone(), current_iterate=best_iterate)
            best_solution = MPPISolution(input_trajectory=best_iterate.input_trajectory,
                                         state_trajectory=solution.state_trajectory,
                                         tracking_cost=solution.tracking_cost)

        solution = best_solution

    save(metadata=self.metadata,
         data={'current_state': current_state.numpy(force=True),
               'goal_state': self.goal_state.numpy(force=True),
               'tracking_cost': solution.tracking_cost.numpy(force=True),
               'input_trajectory': solution.input_trajectory.numpy(force=True),
               'state_trajectory': solution.state_trajectory.numpy(force=True),
               'warm_start_input_trajectory': warm_start_iterate.input_trajectory.numpy(force=True),
               'warm_start_state_trajectory': warm_start_iterate.state_trajectory.numpy(force=True)})

    return solution


def run_mppi(self, current_state, current_iterate):
    current_state = current_state.to(torch.float32)
    current_iterate.input_trajectory = current_iterate.input_trajectory.to(torch.float32)

    self.mpc.U = current_iterate.input_trajectory
    input_trajectory = self.mpc.command(current_state, shift_nominal_trajectory=False)

    state_trajectory = [current_state.unsqueeze(0)]
    for t in range(input_trajectory.shape[0]):
        state_trajectory.append(self.dx(state_trajectory[t], input_trajectory[[t], :]))
    state_trajectory = torch.cat(state_trajectory, dim=-2)

    tracking_cost = self.mpc.running_cost(state_trajectory, None).sum()

    return MPPISolution(state_trajectory=state_trajectory, input_trajectory=input_trajectory,
                        tracking_cost=tracking_cost)
