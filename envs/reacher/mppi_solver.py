import copy

import numpy as np
import torch

from .pytorch_mppi.mppi import MPPI

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation, TimeLimit

from .mppi_utils import MPPIIterate, MPPISolution, select_decorator

from autograd_mujoco import MjStep
from utils import combine_logs_and_delete_temp_files


class EnvDx:
    def __init__(self, env, device: str):
        self.mj_data = copy.deepcopy(env.unwrapped._env.physics.data._data)
        self.mj_model = copy.deepcopy(env.unwrapped._env.physics.model._model)

        self.device = torch.device(device)

        self._state = torch.zeros(self.mj_model.nq + self.mj_model.nv + self.mj_model.na, device=self.device)
        self._ctrl = torch.zeros(self.mj_model.nu, device=self.device)

    def __call__(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        arm = 0.12
        hand = 0.1 + 0.01 * 2

        theta1, theta2, dist_x, dist_y, dtheta1, dtheta2 = obs.split(split_size=1, dim=-1)

        x = arm * torch.cos(theta1) + hand * torch.cos(theta1 + theta2)
        y = arm * torch.sin(theta1) + hand * torch.sin(theta1 + theta2)

        state = torch.cat((theta1, theta2, dtheta1, dtheta2), dim=-1)

        next_state, _, _ = MjStep.apply(state, action, 1, self.mj_model, self.mj_data, 1e-3)

        theta1_next, theta2_next, dtheta1_next, dtheta2_next = next_state.split(split_size=1, dim=-1)

        x_next = arm * torch.cos(theta1_next) + hand * torch.cos(theta1_next + theta2_next)
        y_next = arm * torch.sin(theta1_next) + hand * torch.sin(theta1_next + theta2_next)

        dist_x_next = dist_x + x - x_next
        dist_y_next = dist_y + y - y_next

        next_obs = torch.cat((theta1_next, theta2_next, dist_x_next, dist_y_next, dtheta1_next, dtheta2_next), dim=-1)

        return next_obs

    def unroll_dynamics(self, initial_state: torch.Tensor, input_trajectory: torch.Tensor):
        state = initial_state
        state_trajectory = [state.detach()]
        for input_traj in input_trajectory:
            state = self.__call__(state, input_traj.detach())
            state_trajectory.append(state)
        return torch.stack(state_trajectory, dim=0)

    def state_diff(self, obs1, obs2):
        """
        Compute the difference between two observations, taking into account the periodicity of theta1 and theta2.
        """
        diff = obs1 - obs2
        diff[:, 0] = self.principal_value(diff[:, 0])
        diff[:, 1] = self.principal_value(diff[:, 1])
        return diff

    @staticmethod
    def principal_value(angle):
        """
        Wrap heading angle between -pi and pi.
        """
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def running_cost(obs, action):
    dist_x, dist_y = obs[..., 2], obs[..., 3]
    distances = torch.linalg.norm(torch.stack((dist_x, dist_y), dim=-1), dim=-1)

    reward = -distances
    cost = -reward
    return cost


@select_decorator
class MPPISolver:
    def __init__(self, env, cfg):
        self.metadata = cfg
        self.device = cfg['device']
        self.seed = cfg['seed']

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_device(self.device)

        self.dx = EnvDx(env, self.device)

        goal_target = env.unwrapped.physics.named.model.geom_pos['target', ['x', 'y']]
        cfg['goal_state'] = torch.tensor([*goal_target, 0.0, 0.0, 0.0, 0.0])
        self.goal_state = cfg['goal_state']

        cfg['mpc'].update({
            'noise_sigma': torch.eye(self.dx.mj_model.nu) * torch.tensor(cfg['mpc']['noise_sigma']),
            'u_min': torch.tensor(self.dx.mj_model.actuator_ctrlrange[0][0]),
            'u_max': torch.tensor(self.dx.mj_model.actuator_ctrlrange[0][1]),
            'nx': env.observation_space.shape[0],
            'dynamics': self.dx,
            'running_cost': running_cost,
        })
        self.mpc = MPPI(**cfg['mpc'])

        self.previous_solution = None

    def solve(self, current_state, goal_state=None, current_iterate=None):
        if current_iterate is None:
            current_iterate = self.warm_start_prev_sol()

        self.mpc.U = current_iterate.input_trajectory
        input_trajectory = self.mpc.command(current_state, shift_nominal_trajectory=False)

        state_trajectory = [current_state.unsqueeze(0)]
        for t in range(input_trajectory.shape[0]):
            state_trajectory.append(self.dx(state_trajectory[t], input_trajectory[[t], :]))
        state_trajectory = torch.cat(state_trajectory, dim=-2)

        tracking_cost = self.mpc.running_cost(state_trajectory, None).sum()

        solution = MPPISolution(state_trajectory=state_trajectory, input_trajectory=input_trajectory,
                                tracking_cost=tracking_cost)
        self.previous_solution = solution

        return solution

    def warm_start_prev_sol(self):
        # if the previous solution is None, return a zero input trajectory
        if self.previous_solution is None:
            return MPPIIterate(input_trajectory=torch.zeros(self.mpc.T, self.mpc.nu),
                               state_trajectory=torch.zeros(self.mpc.T + 1, self.mpc.nx))

        # Otherwise, shift and pad with zeros
        input_trajectory = torch.cat((self.previous_solution.input_trajectory[1:], torch.zeros(1, self.mpc.nu)))
        state_trajectory = torch.cat((self.previous_solution.state_trajectory[1:], torch.zeros(1, self.mpc.nx)))

        return MPPIIterate(input_trajectory=input_trajectory, state_trajectory=state_trajectory)


class CustomFlattenObservation(TransformObservation):
    """Defined to allow pickling of the environment (isn't defined with a lambda function)."""
    """Flattens the environment's observation space and each observation from reset and step functions."""

    def __init__(self, env):
        """Constructor for any environment's observation space that implements flatten_space and flatten functions."""
        self.flattened_observation_space = spaces.utils.flatten_space(env.observation_space)
        super().__init__(env, func=self.flatten_observation, observation_space=self.flattened_observation_space)

    def flatten_observation(self, observation):
        """Flatten the observation using the observation space."""
        return self.custom_flatten(self.env.observation_space, observation)

    @staticmethod
    def custom_flatten(observation_space, observation):
        """Custom flatten function for the observation space."""
        if isinstance(observation_space, spaces.Dict):
            return np.concatenate([CustomFlattenObservation.custom_flatten(obs_space, observation[key]) for key, obs_space in observation_space.spaces.items()])
        elif isinstance(observation_space, spaces.Box):
            return observation.flatten()
        else:
            raise NotImplementedError(f"Unsupported observation space: {type(observation_space)}")


class Env:
    def __init__(self, cfg, scenario_token):
        env = DmControlCompatibilityV0(suite.load("reacher", "hard", visualize_reward=True),
                                       render_mode=cfg['env']['render_mode'])
        env = CustomFlattenObservation(env)
        self.env = TimeLimit(env, max_episode_steps=cfg['env']['max_episode_steps'])
        obs, info = self.env.reset(seed=scenario_token)
        self.obs = torch.tensor(obs, dtype=torch.float32)
        cfg['scenario_token'] = scenario_token
        self.solver = MPPISolver(env, cfg)

    def run_simulation(self):
        truncated = False
        obs = self.obs
        while not truncated:
            solution = self.solver.solve(current_state=obs, goal_state=self.solver.goal_state)
            obs, reward, terminated, truncated, info = self.env.step(solution.input_trajectory[0].numpy(force=True))
            obs = torch.from_numpy(obs)

            if self.env.render_mode == "human":
                self.env.render()
        if self.env.render_mode == "human":
            self.env.close()

        combine_logs_and_delete_temp_files(self.solver.metadata)
