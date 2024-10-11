import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from training.model import TransformerModel, TransformerConfig
import wandb


def dynamics(control_limit=1.):
    """
    State update function: x_{k+1} = x_k + u_k
    """
    def next_state(x, u):
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u)
        return x + torch.clamp(u, -control_limit, control_limit)
    return next_state


def cost(a=0.05, b=1.5, c=-2):
    def J(x):
        return (x ** 2 + a) * (x + b) ** 2 * (x + c) ** 2
    return J, (a, b, c)


def find_opt_control_sequence(J, scalars, next_state, x0=0.0, control_limit=1., horizon=5, device=torch.device('cpu')):
    a, b, c = scalars
    # Set the optimal points based on the problem, these are typically the roots of the derivative of J.
    optimal_points = [-b, -c]  # Adjust these points according to where the cost function is minimized
    u_sequences = []  # List to hold control sequences
    x_sequences = []  # List to hold state sequences
    cum_sums = []  # List to hold cumulative costs

    for target in optimal_points:
        u_sequence = []
        x_sequence = [x0]
        x = x0
        remaining_steps = horizon

        while remaining_steps > 0 and abs(x - target) > control_limit:
            step = control_limit if target > x else -control_limit
            u_sequence.append(step)
            x = next_state(x, step)
            x_sequence.append(x)
            remaining_steps -= 1

        # Final step towards the target if within one step's reach
        if remaining_steps > 0:
            final_step = max(min(target - x, control_limit), -control_limit)
            u_sequence.append(final_step)
            x = next_state(x, final_step)
            x_sequence.append(x)
            remaining_steps -= 1

        # Pad with zeros if any steps are left
        u_sequence.extend([0] * remaining_steps)
        x_sequence.extend([x] * remaining_steps)

        u_sequences.append(u_sequence)
        x_sequences.append(x_sequence)

        # Compute cumulative cost
        cum_sum = sum(J(x) for x in x_sequence)
        cum_sums.append(cum_sum)

    return torch.tensor(u_sequences, device=device), torch.tensor(x_sequences, device=device), torch.tensor(cum_sums,
                                                                                                            device=device)


class Loss(torch.nn.Module):
    def __init__(self, ctrl_weight: float, state_weight: float, pairwise_weight: float,
                 miso_method: str, dx):
        super(Loss, self).__init__()
        self.ctrl_weight = ctrl_weight
        self.state_weight = state_weight
        self.pairwise_weight = pairwise_weight
        self.miso_method = miso_method
        self.dx = dx

    def forward(self, predicted_input_trajectory, input_trajectory, state_trajectory):
        B, T, K, _ = predicted_input_trajectory.shape

        input_trajectory = input_trajectory[:, :, None, :].expand(B, T, K, -1)  # [B, T, D] -> [B, T, K, D]
        state_trajectory = state_trajectory[:, :, None, :].expand(B, T + 1, K, -1)  # [B, T, D] -> [B, T, K, D]

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

            diff_state = predicted_state_trajectory[:, 1:, :, :] - state_trajectory[:, 1:, :, :]  # [B, T, K, D]
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
            negative_distances = torch.tanh(-distances)

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


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    """
    Train the model for one epoch
    """
    train_loss, train_ctrl_loss, train_state_loss, train_pd_loss = 0., 0., 0., 0.
    for batch_idx, data in enumerate(dataloader):
        warm_start_input_trajectory, tracking_cost, input_trajectory, state_trajectory = data
        net_inputs = warm_start_input_trajectory
        loss_inputs = {'input_trajectory': input_trajectory, 'state_trajectory': state_trajectory}

        predicted_input_trajectory = model(net_inputs)
        predicted_input_trajectory = F.tanh(predicted_input_trajectory)

        weighted_loss, ctrl_loss, state_loss, pd_loss = \
            loss_fn(predicted_input_trajectory=predicted_input_trajectory, **loss_inputs)

        optimizer.zero_grad()
        weighted_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        with torch.no_grad():
            train_loss += weighted_loss.item()
            train_ctrl_loss += ctrl_loss.item()
            train_state_loss += state_loss.item()
            train_pd_loss += pd_loss.item()

    avg_train_loss = train_loss / len(dataloader)
    avg_train_ctrl_loss = train_ctrl_loss / len(dataloader)
    avg_train_state_loss = train_state_loss / len(dataloader)
    avg_train_pd_loss = train_pd_loss / len(dataloader)
    return avg_train_loss, avg_train_ctrl_loss, avg_train_state_loss, avg_train_pd_loss, grad_norm


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                 train_dataloader):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader

    def train(self, num_epochs: int, start_epoch: int = 1):
        """
        Train the model for num_epochs epochs
        """
        with (tqdm(total=num_epochs - start_epoch + 1) as pbar):
            for epoch in range(start_epoch, num_epochs + 1):
                self.model.train(True)

                avg_train_loss, avg_train_ctrl_loss, avg_train_state_loss, avg_train_pd_loss, grad_norm = \
                    train_one_epoch(self.model, self.train_dataloader, self.optimizer, self.loss_fn)

                pbar.set_description(f'Train Loss: {avg_train_loss:.4f} | Grad Norm: {grad_norm:.4f}')
                pbar.update(1)



def run(config=None, mode='online'):
    with wandb.init(mode=mode, config=config):
        config = wandb.config
        torch.manual_seed(config.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_state = 0
        horizon = 5
        control_limit = 1.
        scalars = (0.05, 1.5, -2)
        next_state = dynamics(control_limit=control_limit)

        J, scalars = cost(*scalars)
        input_trajectory, state_trajectory, tracking_cost = find_opt_control_sequence(J, scalars, next_state, horizon=horizon,
                                                                                      control_limit=control_limit,
                                                                                      device=device)

        data = {'warm_start_input_trajectory': torch.zeros_like(input_trajectory)[:, :, None],
                'tracking_cost': tracking_cost[:, None],
                'input_trajectory': input_trajectory[:, :, None],
                'state_trajectory': state_trajectory[:, :, None]}

        dataset = TensorDataset(*[data[key] for key in data.keys()])
        dataloader = DataLoader(dataset)

        model = TransformerModel(TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=1,
            dropout=0.,
            bias=False,
            is_causal=False,
            src_dim=1,
            src_len=horizon,
            out_dim=1,
            num_predictions=config.num_predictions,
        ))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        loss_fn = Loss(ctrl_weight=config.ctrl_weight,
                       state_weight=config.state_weight,
                       pairwise_weight=config.pairwise_weight,
                       miso_method=config.miso_method,
                       dx=next_state)

        trainer = Trainer(model, optimizer, loss_fn, dataloader)
        trainer.train(int(5e3))

        # Evaluate the model output
        model.eval()
        predicted_input_trajectory = model(torch.zeros(1, 5, 1)).squeeze(0).squeeze(-1).detach()
        predicted_input_trajectory = F.tanh(predicted_input_trajectory)

        # Find the resulting last states
        x = torch.full((1,), current_state, device=device)
        for input_trajectory in predicted_input_trajectory:
            x = next_state(x, input_trajectory)

        # Log the predicted final state to W&B
        wandb.log({f'predicted_x{idx}': x[idx].item() for idx in range(x.shape[0])})
        print("\n Final state:", x.tolist())
        print("Input trajectory:", predicted_input_trajectory.T.tolist())


if __name__ == '__main__':
    wandb_kwargs = {
        'project': 'initpred-illustrative-example',
        'entity': 'crml',
    }
    mode = 'disabled'

    for pairwise_weight, miso_method, num_predictions, seed in \
            ([(0.0, 'none', 1, s) for s in range(5)] +
             [(0.0, 'miso-wta', 2, s) for s in range(5)] +
             [(1.0, 'miso-pd', 2, s) for s in range(5)] +
             [(0.1, 'miso-mix', 2, s) for s in range(5)]):

        run({
            'ctrl_weight': .1,
            'state_weight': 1.,
            'pairwise_weight': pairwise_weight,
            'miso_method': miso_method,
            'seed': seed,
            'num_predictions': num_predictions,
        },
            mode=mode)
