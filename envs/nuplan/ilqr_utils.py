import os
import time
import functools
from dataclasses import replace
from typing import List

import numpy as np

from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import ILQRSolution, ILQRIterate
from utils import combine_logs_and_delete_temp_files, save, repeat_and_perturb, InitPredictor


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
        def wrapped_method(self, current_state, reference_trajectory):
            if not self.init_args_saved_flag:
                self.init_args_saved_flag = True
                save(metadata=self.metadata, data={key: self.metadata[key] for key in ['args', 'kwargs']})
            reference_trajectory_length = reference_trajectory.shape[0]

            warm_start_iterate = warm_start_prev_sol(self=self, current_state=current_state,
                                                     reference_trajectory_length=reference_trajectory_length)

            solution_list = solve(self=self, current_state=current_state,
                                  reference_trajectory=reference_trajectory, warm_start_iterate=warm_start_iterate)

            self.previous_solution = ILQRIterate(
                input_trajectory=solution_list[-1].input_trajectory,
                state_trajectory=solution_list[-1].state_trajectory,
                input_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_inputs)),
                state_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_states)),
            )

            self.metadata['iteration'] += 1

            return solution_list

        return wrapped_method

    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_solve_time = self.metadata['solver']['max_solve_time']
            self.max_ilqr_iterations = self.metadata['solver']['max_ilqr_iterations']
            self.method = self.metadata['method']
            self.metadata['iteration'] = 0
            # log args and kwargs
            self.metadata['args'] = args
            self.metadata['kwargs'] = kwargs
            self.init_args_saved_flag = self.method != 'warm_start'  # whether the init args have been saved
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
            current_state = self.ilqr_log['current_state'][self.metadata['iteration']]
            reference_trajectory = self.ilqr_log['reference_trajectory'][self.metadata['iteration']]

            reference_trajectory_length = reference_trajectory.shape[0]

            warm_start_iterate = ILQRIterate(
                input_trajectory=self.ilqr_log['warm_start_input_trajectory'][self.metadata['iteration']],
                state_trajectory=self.ilqr_log['warm_start_state_trajectory'][self.metadata['iteration']],
                input_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_inputs)),
                state_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_states)),
            )

            solution_list = solve(self=self, current_state=current_state,
                                  reference_trajectory=reference_trajectory, warm_start_iterate=warm_start_iterate)

            self.metadata['iteration'] += 1

            return True

        return wrapped_method

    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            load_filename = f"{args[0]['load_path']}{args[0]['load_name']}_{args[0]['scenario_token']}"
            solver_init = np.load(f"{load_filename}_solver_init.npz", allow_pickle=True)
            solver_params = solver_init['kwargs'].item()['solver_params']
            solver_params = replace(solver_params, metadata=args[0])
            warm_start_params = solver_init['kwargs'].item()['warm_start_params']
            super().__init__(solver_params=solver_params, warm_start_params=warm_start_params)
            self.max_solve_time = self.metadata['solver']['max_solve_time']
            self.max_ilqr_iterations = self.metadata['solver']['max_ilqr_iterations']
            self.method = self.metadata['method']
            self.metadata['iteration'] = 0
            self.ilqr_log = np.load(f"{load_filename}.npz", allow_pickle=True)
            self.metadata['iterations'] = self.ilqr_log['tracking_cost'].shape[0]

            if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
                self.predict = InitPredictor(self.metadata)

            # Wrap specified methods.yaml
            for method_name in ['solve']:
                if hasattr(self, method_name):
                    original_method = getattr(self, method_name)
                    wrapped = wrap_method(original_method)
                    setattr(self, method_name, wrapped.__get__(self, type(self)))

    return WrappedClass


def solve(self, current_state, reference_trajectory, warm_start_iterate):
    if self.method in ['warm_start', 'oracle']:
        solution_list = run_ilqr(self=self, current_state=current_state,
                                 reference_trajectory=reference_trajectory,
                                 current_iterate=warm_start_iterate)
    elif self.method in ['NN', 'NN_perturbation', 'NN_ensemble', 'warm_start_perturbation']:
        if self.method in ['NN', 'NN_perturbation', 'NN_ensemble']:
            predicted_input_trajectory = self.predict(reference_trajectory=reference_trajectory,
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
            current_iterate = self._run_forward_dynamics(current_state, input_trajectory)

            if self.metadata['optimizer_mode'] == 'multiple':
                solution_list = run_ilqr(self=self, current_state=current_state,
                                         reference_trajectory=reference_trajectory,
                                         current_iterate=current_iterate)
                last_solution = solution_list[-1]

                if last_solution.tracking_cost < lowest_cost:
                    lowest_cost = last_solution.tracking_cost
                    best_solution = ILQRSolution(input_trajectory=last_solution.input_trajectory,
                                                 state_trajectory=last_solution.state_trajectory,
                                                 tracking_cost=last_solution.tracking_cost)

            elif self.metadata['optimizer_mode'] == 'single':
                tracking_cost = self._compute_tracking_cost(
                    iterate=current_iterate, reference_trajectory=reference_trajectory)
                if tracking_cost < lowest_cost:
                    lowest_cost = tracking_cost
                    best_iterate = current_iterate

        if self.metadata['optimizer_mode'] == 'single':
            solution_list = run_ilqr(self=self, current_state=current_state,
                                     reference_trajectory=reference_trajectory,
                                     current_iterate=best_iterate)
            last_solution = solution_list[-1]
            best_solution = ILQRSolution(input_trajectory=last_solution.input_trajectory,
                                         state_trajectory=last_solution.state_trajectory,
                                         tracking_cost=last_solution.tracking_cost)

        solution_list = [best_solution]

    save(metadata=self.metadata,
         data={'current_state': current_state,
               'reference_trajectory': reference_trajectory,
               'num_iterations': len(solution_list),
               'tracking_cost': solution_list[-1].tracking_cost,
               'input_trajectory': solution_list[-1].input_trajectory,
               'state_trajectory': solution_list[-1].state_trajectory,
               'warm_start_input_trajectory': warm_start_iterate.input_trajectory,
               'warm_start_state_trajectory': warm_start_iterate.state_trajectory})

    return solution_list


def run_ilqr(self, current_state, reference_trajectory, current_iterate):
    """
    Run the iLQR solver.
    """
    solution_list: List[ILQRSolution] = []

    solve_start_time = time.perf_counter()
    for _ in range(self.max_ilqr_iterations):
        # Determine the cost and store the associated solution object.
        tracking_cost = self._compute_tracking_cost(
            iterate=current_iterate,
            reference_trajectory=reference_trajectory,
        )
        solution_list.append(
            ILQRSolution(
                input_trajectory=current_iterate.input_trajectory,
                state_trajectory=current_iterate.state_trajectory,
                tracking_cost=tracking_cost,
            )
        )

        # Determine the LQR optimal perturbations to apply.
        lqr_input_policy = self._run_lqr_backward_recursion(
            current_iterate=current_iterate,
            reference_trajectory=reference_trajectory,
        )

        # Apply the optimal perturbations to generate the next input trajectory iterate.
        input_trajectory_next = self._update_inputs_with_policy(
            current_iterate=current_iterate,
            lqr_input_policy=lqr_input_policy,
        )

        current_iterate = self._run_forward_dynamics(current_state, input_trajectory_next)

        elapsed_time = time.perf_counter() - solve_start_time
        if elapsed_time >= self.max_solve_time:
            break

    # Store the final iterate in the solution_dict.
    tracking_cost = self._compute_tracking_cost(
        iterate=current_iterate,
        reference_trajectory=reference_trajectory
    )
    solution_list.append(
        ILQRSolution(
            input_trajectory=current_iterate.input_trajectory,
            state_trajectory=current_iterate.state_trajectory,
            tracking_cost=tracking_cost,
        )
    )

    return solution_list


def warm_start_prev_sol(self, current_state, reference_trajectory_length):
    """
    Warm start the with the previous solution by shifting the previous input and padding with zeros
    """
    # if the previous solution is None, return a zero input trajectory
    if self.previous_solution is None:
        return ILQRIterate(
            input_trajectory=np.zeros((reference_trajectory_length - 1, self._n_inputs)),
            state_trajectory=np.zeros((reference_trajectory_length, self._n_states)),
            input_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_inputs)),
            state_jacobian_trajectory=np.zeros((reference_trajectory_length - 1, self._n_states, self._n_states)),
        )

    # Otherwise, shift and pad with zeros
    previous_input_trajectory_shifted = np.roll(self.previous_solution.input_trajectory, -1, axis=0)
    previous_input_trajectory_shifted[-1, :] = 0

    # match the length of the previous_input_trajectory_shifted with reference_trajectory_length if it is longer
    # or pad with zeros if it is shorter
    if previous_input_trajectory_shifted.shape[0] > reference_trajectory_length - 1:
        previous_input_trajectory_shifted = previous_input_trajectory_shifted[:reference_trajectory_length - 1, :]
    elif previous_input_trajectory_shifted.shape[0] < reference_trajectory_length - 1:
        previous_input_trajectory_shifted = np.pad(previous_input_trajectory_shifted,
                                                   ((0, reference_trajectory_length - 1 -
                                                     previous_input_trajectory_shifted.shape[0]),
                                                    (0, 0)),
                                                   mode='constant')

    return self._run_forward_dynamics(current_state,
                                      previous_input_trajectory_shifted[:reference_trajectory_length - 1, :])
