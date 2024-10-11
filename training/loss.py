import torch
from training.envs_dx import get_dx


class Loss(torch.nn.Module):
    def __init__(self, ctrl_weight: float, state_weight: float, pairwise_weight: float,
                 miso_method: str, env: str, scaler: dict):
        super(Loss, self).__init__()
        self.ctrl_weight = ctrl_weight
        self.state_weight = state_weight
        self.pairwise_weight = pairwise_weight
        self.miso_method = miso_method
        self.dx = get_dx(env)
        self.scaler = scaler

    def forward(self, _predicted_input_trajectory, input_trajectory, state_trajectory):
        B, T, K, _ = _predicted_input_trajectory.shape

        predicted_input_trajectory = self.inverse_transform(_predicted_input_trajectory, key='input_trajectory')
        input_trajectory = input_trajectory[:, :, None, :].expand(B, T, K, -1)  # [B, T, D] -> [B, T, K, D]
        state_trajectory = state_trajectory[:, :, None, :].expand(B, T + 1, K, -1) # [B, T, D] -> [B, T, K, D]

        ctrl_loss = torch.mean(torch.square(predicted_input_trajectory - input_trajectory), dim=(-3, -1))  # [B, T, K, D] -> [B, K]
        weighted_loss = self.ctrl_weight * ctrl_loss

        if self.state_weight:
            predicted_input_trajectory = predicted_input_trajectory.permute(0, 2, 1, 3).flatten(0, 1)  # [B, T, K, D] -> [B*K, T, D]
            current_state = state_trajectory[:, 0, :, :].flatten(0, 1)  # [B, T, K, D] -> [B, K, D] -> [B*K, D]

            predicted_state_trajectory = [current_state]
            for t in range(T):
                predicted_state_trajectory.append(self.dx(predicted_state_trajectory[t], predicted_input_trajectory[:, t, :]))
            predicted_state_trajectory = torch.stack(predicted_state_trajectory, dim=1)  # [B*K, T+1, D]

            predicted_input_trajectory = predicted_input_trajectory.unflatten(0, (B, K)).permute(0, 2, 1, 3)  # [B*K, T, D] -> [B, T, K, D]
            predicted_state_trajectory = predicted_state_trajectory.unflatten(0, (B, K))  # [B*K, T+1, D] -> [B, K, T+1, D]
            predicted_state_trajectory = predicted_state_trajectory.permute(0, 2, 1, 3)  # [B, K, T+1, D] -> [B, T, K, D]

            diff_state = self.dx.state_diff(predicted_state_trajectory[:, 1:, :, :], state_trajectory[:, 1:, :, :])  # [B, T, K, D]
            state_loss = torch.mean(torch.square(diff_state), dim=(-3, -1))  # [B, T, K, D] -> [B, K]

        else:
            state_loss = torch.zeros_like(ctrl_loss)

        weighted_loss += self.state_weight * state_loss

        if self.miso_method in ['miso-pd', 'miso-mix']:
            outputs_exp = predicted_input_trajectory.unsqueeze(2) - predicted_input_trajectory.unsqueeze(3)   # [B, T, K, K, D]
            # Calculate mean distance for each mode. Self-distance will be always zero.
            # Average will be over K instead of K-1, but it shouldn't matter much.
            distances = torch.norm(outputs_exp, dim=-1)  # [B, T, K, K, D] -> [B, T, K, K]
            distances = distances.mean(dim=1)  # [B, T, K, K] -> [B, K, K]
            # Apply soft-clipping to pair-wise distances
            negative_distances = torch.tanh(-distances * 0.1)

            # Average over pairs
            pd_loss = negative_distances.mean(dim=2)  # [B, K, K] -> [B, K]
        else:
            pd_loss = torch.zeros_like(ctrl_loss)

        weighted_loss += self.pairwise_weight * pd_loss

        if self.miso_method in ['miso-wta', 'miso-mix']:
            # Pick only the best guess based on the combined loss.
            best_mode = torch.argmin(weighted_loss, dim=1)
            # Use the best mode to index the K dimension of loss. [B, K] -> [B, 1]
            weighted_loss = torch.gather(weighted_loss, 1, best_mode[:, None])
            ctrl_loss = torch.gather(ctrl_loss, 1, best_mode[:, None])
            state_loss = torch.gather(state_loss, 1, best_mode[:, None])
            pd_loss = torch.gather(pd_loss, 1, best_mode[:, None])

        # Return mean over batch and modes
        return weighted_loss.mean(), ctrl_loss.mean(), state_loss.mean(), pd_loss.mean()

    def inverse_transform(self, x, key: str):
        return x * self.scaler[key]['scale_'] + self.scaler[key]['mean_']

    def transform(self, x, key: str):
        return (x - self.scaler[key]['mean_']) / self.scaler[key]['scale_']
