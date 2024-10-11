"""
This provides an implementation of the iterative linear quadratic regulator (iLQR) algorithm for trajectory tracking.
It is specialized to the case with a discrete-time kinematic bicycle model and a quadratic trajectory tracking cost.

Original (Nonlinear) Discrete Time System:
    z_k = [x_k, y_k, theta_k, v_k, delta_k]
    u_k = [a_k, phi_k]

    x_{k+1}     = x_k     + v_k * cos(theta_k) * dt
    y_{k+1}     = y_k     + v_k * sin(theta_k) * dt
    theta_{k+1} = theta_k + v_k * tan(delta_k) / L * dt
    v_{k+1}     = v_k     + a_k * dt
    delta_{k+1} = delta_k + phi_k * dt

    where (x_k, y_k, theta_k) is the pose at timestep k with time discretization dt,
    v_k and a_k are velocity and acceleration,
    delta_k and phi_k are steering angle and steering angle rate,
    and L is the vehicle wheelbase.

Quadratic Tracking Cost:
    J = sum_{k=0}^{N-1} ||u_k||_2^{R_k} +
        sum_{k=0}^N ||z_k - z_{ref,k}||_2^{Q_k}
For simplicity, we opt to use constant input cost matrices R_k = R and constant state cost matrices Q_k = Q.

There are multiple improvements that can be done for this implementation, but omitted for simplicity of the code.
Some of these include:
  * Handle constraints directly in the optimization (e.g. log-barrier / penalty method with quadratic cost estimate).
  * Line search in the input policy update (feedforward term) to determine a good gradient step size.

References Used: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec5-LQR.pdf and
                 https://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_trajectoryoptimization.pdf
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    complete_kinematic_state_and_inputs_from_poses,
    compute_steering_angle_feedback,
)

from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver, ILQRSolution, ILQRIterate)

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from .ilqr_utils import select_decorator

DoubleMatrix = npt.NDArray[np.float64]


@select_decorator
class ModifiedILQRSolver(ILQRSolver):
    """
    Modified ILQRSolver to implement warm start.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_solution = None
        self.metadata = kwargs['solver_params'].metadata

    def solve(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> List[ILQRSolution]:
        """
        Run the main iLQR loop used to try to find (locally) optimal inputs to track the reference trajectory.
        :param current_state: The initial state from which we apply inputs, z_0.
        :param reference_trajectory: The state reference we'd like to track, inclusive of the initial timestep,
                                     z_{r,k} for k in {0, ..., N}.
        :return: A list of solution iterates after running the iLQR algorithm where the index is the iteration number.
        """
        # Check that state parameter has the right shape.
        assert current_state.shape == (self._n_states,), "Incorrect state shape."

        # Check that reference trajectory parameter has the right shape.
        assert len(reference_trajectory.shape) == 2, "Reference trajectory should be a 2D matrix."
        reference_trajectory_length, reference_trajectory_state_dimension = reference_trajectory.shape
        assert reference_trajectory_length > 1, "The reference trajectory should be at least two timesteps long."
        assert (
            reference_trajectory_state_dimension == self._n_states
        ), "The reference trajectory should have a matching state dimension."

        # List of ILQRSolution results where the index corresponds to the iteration of iLQR.
        solution_list: List[ILQRSolution] = []

        # Get warm start input and state trajectory, as well as associated Jacobians.
        # current_iterate = self._input_warm_start(current_state, reference_trajectory)
        self.previous_solution = ILQRIterate(
            input_trajectory=np.zeros((reference_trajectory_length-1, self._n_inputs)),
            state_trajectory=np.zeros((reference_trajectory_length, self._n_states)),
            input_jacobian_trajectory=np.zeros((reference_trajectory_length-1, self._n_states, self._n_inputs)),
            state_jacobian_trajectory=np.zeros((reference_trajectory_length-1, self._n_states, self._n_states)),
        ) if self.previous_solution is None else self.previous_solution
        current_iterate = self._warm_start_prev_sol(current_state, self.previous_solution, reference_trajectory_length)

        # Main iLQR Loop.
        solve_start_time = time.perf_counter()
        for _ in range(self._solver_params.max_ilqr_iterations):
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

            # Check for convergence/timeout and terminate early if so.
            # Else update the input_trajectory iterate and continue.
            input_trajectory_norm_difference = np.linalg.norm(input_trajectory_next - current_iterate.input_trajectory)

            current_iterate = self._run_forward_dynamics(current_state, input_trajectory_next)

            if input_trajectory_norm_difference < self._solver_params.convergence_threshold:
                break

            elapsed_time = time.perf_counter() - solve_start_time
            if isinstance(self._solver_params.max_solve_time, float) and elapsed_time >= self._solver_params.max_solve_time:
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
        self.previous_solution = ILQRIterate(
            input_trajectory=current_iterate.input_trajectory,
            state_trajectory=current_iterate.state_trajectory,
            input_jacobian_trajectory=np.zeros((reference_trajectory_length-1, self._n_states, self._n_inputs)),
            state_jacobian_trajectory=np.zeros((reference_trajectory_length-1, self._n_states, self._n_states)),
        )

        return solution_list

    def _input_warm_start(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> ILQRIterate:
        """
        Given a reference trajectory, we generate the warm start (initial guess) by inferring the inputs applied based
        on poses in the reference trajectory.
        :param current_state: The initial state from which we apply inputs.
        :param reference_trajectory: The reference trajectory we are trying to follow.
        :return: The warm start iterate from which to start iLQR.
        """
        reference_states_completed, reference_inputs_completed = complete_kinematic_state_and_inputs_from_poses(
            discretization_time=self._solver_params.discretization_time,
            wheel_base=self._solver_params.wheelbase,
            poses=reference_trajectory[:, :3],
            jerk_penalty=self._warm_start_params.jerk_penalty_warm_start_fit,
            curvature_rate_penalty=self._warm_start_params.curvature_rate_penalty_warm_start_fit,
        )

        # We could just stop here and apply reference_inputs_completed (assuming it satisfies constraints).
        # This could work if current_state = reference_states_completed[0,:] - i.e. no initial tracking error.
        # We add feedback input terms for the first control input only to account for nonzero initial tracking error.
        _, _, _, velocity_current, steering_angle_current = current_state
        _, _, _, velocity_reference, steering_angle_reference = reference_states_completed[0, :]

        acceleration_feedback = -self._warm_start_params.k_velocity_error_feedback * (
                velocity_current - velocity_reference
        )

        steering_angle_feedback = compute_steering_angle_feedback(
            pose_reference=current_state[:3],
            pose_current=reference_states_completed[0, :3],
            lookahead_distance=self._warm_start_params.lookahead_distance_lateral_error,
            k_lateral_error=self._warm_start_params.k_lateral_error,
        )
        steering_angle_desired = steering_angle_feedback + steering_angle_reference
        steering_rate_feedback = -self._warm_start_params.k_steering_angle_error_feedback * (
                steering_angle_current - steering_angle_desired
        )

        reference_inputs_completed[0, 0] += acceleration_feedback
        reference_inputs_completed[0, 1] += steering_rate_feedback

        # We rerun dynamics with constraints applied to make sure we have a feasible warm start for iLQR.
        return self._run_forward_dynamics(current_state, reference_inputs_completed)

    def _warm_start_prev_sol(self, current_state: DoubleMatrix, previous_solution: ILQRSolution,
                             reference_trajectory_length: int) -> ILQRIterate:
        """
        Given the previous solution, we generate the warm start (initial guess) by inferring the inputs applied based
        on the previous solution.
        :param current_state: The initial state from which we apply inputs.
        :param previous_solution: The previous solution we are trying to follow.
        :return: The warm start iterate from which to start iLQR.
        """
        # shift and pad with zeros
        previous_input_trajectory_shifted = np.roll(previous_solution.input_trajectory, -1, axis=0)
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


@dataclass(frozen=True)
class ModifiedILQRSolverParameters:
    """ Same as ILQRSolverParameters but with additional parameters for metadata."""
    """Parameters related to the solver implementation."""
    metadata: Optional[Dict[str, Any]]  # Metadata containing the scenario token and possibly other relevant information.

    discretization_time: float  # [s] Time discretization used for integration.

    # Cost weights for state [x, y, heading, velocity, steering angle] and input variables [acceleration, steering rate].
    state_cost_diagonal_entries: List[float]
    input_cost_diagonal_entries: List[float]

    # Trust region cost weights for state and input variables.  Helps keep linearization error per update step bounded.
    state_trust_region_entries: List[float]
    input_trust_region_entries: List[float]

    # Parameters related to solver runtime / solution sub-optimality.
    max_ilqr_iterations: int  # Maximum number of iterations to run iLQR before timeout.
    convergence_threshold: float  # Threshold for delta inputs below which we can terminate iLQR early.
    max_solve_time: Optional[
        float
    ]  # [s] If defined, sets a maximum time to run a solve call of iLQR before terminating.

    # Constraints for underlying dynamics model.
    max_acceleration: float  # [m/s^2] Absolute value threshold on acceleration input.
    max_steering_angle: float  # [rad] Absolute value threshold on steering angle state.
    max_steering_angle_rate: float  # [rad/s] Absolute value threshold on steering rate input.

    # Parameters for dynamics / linearization.
    min_velocity_linearization: float  # [m/s] Absolute value threshold below which linearization velocity is modified.
    wheelbase: float = get_pacifica_parameters().wheel_base  # [m] Wheelbase length parameter for the vehicle.

    def __post_init__(self) -> None:
        """Ensure entries lie in expected bounds and initialize wheelbase."""
        for entry in [
            "discretization_time",
            "max_ilqr_iterations",
            "convergence_threshold",
            "max_acceleration",
            "max_steering_angle",
            "max_steering_angle_rate",
            "min_velocity_linearization",
            "wheelbase",
        ]:
            assert getattr(self, entry) > 0.0, f"Field {entry} should be positive."

        assert self.max_steering_angle < np.pi / 2.0, "Max steering angle should be less than 90 degrees."

        if isinstance(self.max_solve_time, float):
            assert self.max_solve_time > 0.0, "The specified max solve time should be positive."

        assert np.all([x >= 0 for x in self.state_cost_diagonal_entries]), "Q matrix must be positive semidefinite."
        assert np.all([x > 0 for x in self.input_cost_diagonal_entries]), "R matrix must be positive definite."

        assert np.all(
            [x > 0 for x in self.state_trust_region_entries]
        ), "State trust region cost matrix must be positive definite."
        assert np.all(
            [x > 0 for x in self.input_trust_region_entries]
        ), "Input trust region cost matrix must be positive definite."


