import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import mujoco as mj
from mujoco import rollout


class MjStep(Function):
    """
    A custom autograd function for the MuJoCo step function.
    This is required because the MuJoCo step function is not differentiable.
    """

    @staticmethod
    def forward(*args, **kwargs) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Forward pass of the MjStep function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            next_state: The next state after the step.
            dydx: The derivative of y with respect to x.
            dydu: The derivative of y with respect to u.

        Dimensions key:
        B: Batch size
        nq: Number of position variables
        nv: Number of velocity variables
        na: Number of actuator variables
        nu: Number of control variables
        n_steps: Number of steps in the rollout

        state.shape = [B, nq + nv + na]
        ctrl.shape = [B, 1, nu]

        """
        # Extracting the arguments
        state, ctrl, n_steps, mj_model, mj_data, eps = args
        dydx, dydu = [], []
        device = state.device
        compute_grads = state.requires_grad or ctrl.requires_grad

        state, ctrl = state.numpy(force=True), ctrl.numpy(force=True)

        # Repeat the control input for each step in the rollout: [B, 1, nu] -> [B, n_steps, nu]
        ctrl = np.repeat(ctrl[:, None, :], n_steps, axis=1)

        if state.shape[-1] == mj.mj_stateSize(mj_model, mj.mjtState.mjSTATE_FULLPHYSICS.value) - 1:
            """
            As of MuJoCo 3.1.2 the initial state passed to rollout() must include a time step. 
            Manually concatenating a time-step to the state vector.
            """
            state = np.concatenate([np.zeros_like(state[:, [0]]), state], axis=1)

        # Perform the rollout and get the states
        states, _ = mj.rollout.rollout(mj_model, mj_data, state, ctrl)

        # Pop the time-step from state and states (we don't need it for the gradients)
        state, states = state[:, 1:], states[:, :, 1:]

        # Reshape the states and control inputs: [B, n_steps, nq + nv + na], [B, n_steps, nu]
        states = states.reshape(-1, n_steps, mj_model.nq + mj_model.nv + mj_model.na)
        ctrl = ctrl.reshape(-1, n_steps, mj_model.nu)

        if compute_grads:
            # Concatenate the initial state with the rest of the states
            _states = np.concatenate([state[:, None, :], states[:, :-1, :]], axis=1)
            # Set the solver tolerances and disable the warmstart for the solver
            mj_model.opt.tolerance = 0  # Disable early termination to make the same number of steps for each FD call
            mj_model.opt.ls_tolerance = 1e-18  # Set the line search tolerance to a very low value, for stability
            mj_model.opt.disableflags = 2 ** 8  # Disable solver warmstart
            for (state_batch, ctrl_batch) in zip(_states, ctrl):
                # Initialize the A and B matrices for the approximated linear system
                A = np.eye(2 * mj_model.nv + mj_model.na)
                B = np.zeros((2 * mj_model.nv + mj_model.na, mj_model.nu))

                for (_state, _ctrl) in zip(state_batch, ctrl_batch):
                    # Reset the MuJoCo data and set the state and control
                    mj.mj_resetData(mj_model, mj_data)
                    MjStep.set_state_and_ctrl(mj_model, mj_data, _state, _ctrl)

                    # Initialize the _A and _B matrices for the approximated linear system
                    _A = np.zeros((2 * mj_model.nv + mj_model.na, 2 * mj_model.nv + mj_model.na))
                    _B = np.zeros((2 * mj_model.nv + mj_model.na, mj_model.nu))

                    # Compute the forward dynamics using MuJoCo's built-in function
                    mj.mjd_transitionFD(mj_model, mj_data, eps, 0, _A, _B, None, None)

                    # Update the A and B matrices
                    A = np.matmul(_A, A)
                    B = np.matmul(_A, B) + _B

                # Append the A and B matrices to the lists
                dydx.append(A.copy())
                dydu.append(B.copy())

            # Reset the solver tolerances and enable the warmstart for the solver
            mj_model.opt.tolerance = 1e-10
            mj_model.opt.ls_tolerance = 1e-10
            mj_model.opt.disableflags = 0

        # Get the final state from the rollout
        next_state = torch.as_tensor(states[:, -1, :], device=device, dtype=torch.float32)

        # Convert the lists of A and B matrices to numpy arrays
        dydx = np.array(dydx) if compute_grads else None
        dydu = np.array(dydu) if compute_grads else None

        return next_state, dydx, dydu

    @staticmethod
    def setup_context(ctx: Function, inputs: tuple, output: tuple) -> None:
        """
        Set up the context for the backward pass.
        """
        state, _, _, _, _, _ = inputs
        _, dydx, dydu = output

        dydx = torch.from_numpy(dydx.astype(np.float32)).to(state.device) if dydx is not None else None
        dydu = torch.from_numpy(dydu.astype(np.float32)).to(state.device) if dydu is not None else None

        ctx.save_for_backward(dydx, dydu)

    @staticmethod
    @once_differentiable
    def backward(ctx: Function, grad_output: torch.Tensor, *args) \
            -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        """
        Backward pass of the MjStep function.

        Args:
            ctx: The context object where results are saved for backward computation.
            grad_output: The output of the forward method.
            *args: Variable length argument list.

        Returns:
            grad_state: The gradient of the state.
            grad_ctrl: The gradient of the control.
            None, None, None, None: Placeholder for other gradients that are not computed.
        """
        dydx, dydu = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            # dL/dx = dL/dy * dy/dx ([B, 1, nq + nv] * [B, nq + nv, nq + nv] = [B, 1, nq + nv])
            grad_state = torch.bmm(grad_output[:, None, :], dydx).squeeze(1)
        else:
            grad_state = None

        if ctx.needs_input_grad[1]:
            # dL/du = dL/dy * dy/du ([B, 1, nq + nv] * [B, nq + nv, 1] = [B, 1, 1])
            grad_ctrl = torch.bmm(grad_output[:, None, :], dydu).squeeze(1)

        else:
            grad_ctrl = None

        return grad_state, grad_ctrl, None, None, None, None

    @staticmethod
    def set_state_and_ctrl(mj_model: mj.MjModel, mj_data: mj.MjData, state: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Set the state and control for the MuJoCo model.
        """
        mj.mj_resetData(mj_model, mj_data)
        np.copyto(mj_data.qpos, state[:mj_model.nq].squeeze())
        np.copyto(mj_data.qvel, state[mj_model.nq:mj_model.nq + mj_model.nv].squeeze())
        np.copyto(mj_data.ctrl, ctrl.squeeze())

    @staticmethod
    def get_state(mj_data: mj.MjData) -> np.ndarray:
        """
        Get the state from the MuJoCo data.
        """
        return np.concatenate([mj_data.qpos, mj_data.qvel])
