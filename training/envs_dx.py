import torch


def get_dx(env):
    if env == 'nuplan':
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
        vehicle = get_pacifica_parameters()

        class KinematicBicycleModel(torch.nn.Module):
            def __init__(self,
                         vehicle: VehicleParameters,
                         max_steering_angle=1.047197,  # [rad] Absolute value threshold on steering angle.
                         max_acceleration=3.0,  # [m/s^2] Absolute value threshold on acceleration input.
                         max_steering_angle_rate=0.5,  # [rad/s] Absolute value threshold on steering rate input.
                         discretization_time=0.2,  # [s] Time discretization used for integration.
                         device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                         ):
                super(KinematicBicycleModel, self).__init__()
                self.device = device
                self._vehicle_wheel_base = torch.tensor(vehicle.wheel_base, device=self.device)
                self._max_steering_angle = torch.tensor(max_steering_angle, device=self.device)
                self._input_clip_min = torch.tensor((-max_acceleration, -max_steering_angle_rate), device=self.device)
                self._input_clip_max = torch.tensor((max_acceleration, max_steering_angle_rate), device=self.device)
                self._discretization_time = torch.tensor(discretization_time, device=self.device)

            def forward(self, current_state, current_input):
                """
                Propagates the state forward by one step.
                We also impose all constraints here to ensure the current input and next state are always feasible.
                :param current_state: The current state z_k.
                :param current_input: The applied input u_k.
                :return: The next state z_{k+1}.
                """

                x, y, heading, velocity, steering_angle = current_state.split(split_size=1, dim=-1)

                # Input constraints: clip inputs within bounds and then use.
                current_input = self._clip_inputs(current_input)
                acceleration, steering_rate = current_input.split(split_size=1, dim=-1)

                # Create a new tensor for next_state to avoid inplace modifications on current_state
                next_state = current_state.clone()

                # Euler integration of bicycle model, performed out-of-place
                delta_x = velocity * torch.cos(heading) * self._discretization_time
                delta_y = velocity * torch.sin(heading) * self._discretization_time
                delta_heading = velocity * torch.tan(
                    steering_angle) / self._vehicle_wheel_base * self._discretization_time
                delta_velocity = acceleration * self._discretization_time
                delta_steering_angle = steering_rate * self._discretization_time

                # Update next_state using out-of-place operations
                next_state = next_state + torch.cat(
                    [delta_x, delta_y, delta_heading, delta_velocity, delta_steering_angle], dim=-1)

                # Constrain heading angle to lie within +/- pi.
                constrained_heading = self._principal_value(next_state[:, [2]])

                # State constraints: clip the steering_angle within bounds and update steering_rate accordingly.
                constrained_steering_angle = self._clip_steering_angle(next_state[:, [4]])

                # Reassemble next_state with constrained values without losing gradients
                next_state = torch.cat([
                    next_state[:, :2],  # x and y
                    constrained_heading,  # constrained heading
                    next_state[:, 3:4],  # velocity
                    constrained_steering_angle  # constrained steering angle
                ], dim=-1)

                return next_state

            def _clip_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
                """
                Clip the inputs to lie within the valid range.
                """
                return torch.clamp(inputs, self._input_clip_min, self._input_clip_max)

            def _clip_steering_angle(self, steering_angle: torch.Tensor) -> torch.Tensor:
                """
                Clip the steering angle to lie within the valid range.
                """
                return torch.clamp(steering_angle, -self._max_steering_angle, self._max_steering_angle)

            @staticmethod
            def _principal_value(angle):
                """
                Wrap heading angle between -pi and pi.
                """
                return (angle + torch.pi) % (2 * torch.pi) - torch.pi

            def state_diff(self, state1, state2):
                """
                Compute the difference between two states, taking into account the periodicity of the heading and steering angle.
                """
                diff = state1 - state2
                diff[:, 3] = diff[:, 3].detach()  # detach velocity (gradients seem to be causing issues)
                diff[:, 2] = self._principal_value(diff[:, 2])
                diff[:, 4] = self._principal_value(diff[:, 4])
                return diff

        dx = KinematicBicycleModel(vehicle, device=torch.device('cuda'))

    elif env in ['reacher', 'cartpole']:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(f"envs/{env}/config.yaml")
        cfg = OmegaConf.to_container(cfg, resolve=True)

        if env == 'reacher':
            from envs.reacher.mppi_solver import EnvDx, CustomFlattenObservation
            from shimmy.dm_control_compatibility import DmControlCompatibilityV0
            from dm_control import suite
            from gymnasium.wrappers import TimeLimit

            environment = DmControlCompatibilityV0(suite.load("reacher", "hard", visualize_reward=True),
                                                   render_mode='rgb_array')
            environment = CustomFlattenObservation(environment)
            environment = TimeLimit(environment, max_episode_steps=int(cfg['env']['max_episode_steps']))

            dx = EnvDx(environment, 'cpu')

        elif env == 'cartpole':
            from envs.cartpole.boxddp_solver import EnvDx

            dx = EnvDx(**{k: v for k, v in cfg['env'].items() if k != 'T_env'})

    else:
        raise ValueError("Invalid environment")

    return dx
