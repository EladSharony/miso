import os
import pickle

import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset, DataLoader

from tqdm import tqdm
import wandb
from utils import get_wandb_runs

from training.model import TransformerModel, TransformerConfig
from training.loss import Loss


def handle_data(data: tuple):
    """ The only difference is that in nuplan it's a reference trajectory instead of a goal state (the math is the same)"""
    current_state, reference_trajectory, input_trajectory, state_trajectory, warm_start_input_trajectory, warm_start_state_trajectory, _current_state, _reference_trajectory, _input_trajectory, _state_trajectory, _warm_start_input_trajectory, _warm_start_state_trajectory = data

    _tracking_error = (_reference_trajectory - _warm_start_state_trajectory)[:, :-1, :]

    net_inputs = torch.cat((_tracking_error, _warm_start_input_trajectory), dim=-1)
    loss_inputs = {'input_trajectory': input_trajectory, 'state_trajectory': state_trajectory}
    return net_inputs, loss_inputs


def prepare_dataloaders(dataset: TensorDataset, cfg: dict):
    """
    Prepare the dataloader for the distributed training
    """
    # Split the dataset into train and validation sets
    train_set, val_set = random_split(dataset, [.8, .2], generator=torch.Generator().manual_seed(cfg['train']["seed"]))
    kwargs = {'batch_size': cfg['train']["batch_size"], 'pin_memory': False, 'shuffle': True}

    train_loader = DataLoader(train_set, **kwargs)
    val_loader = DataLoader(val_set, **kwargs)

    return train_loader, val_loader


def load_trainer_objs(cfg: dict):
    data = torch.load(f"{cfg['data_path']}/{cfg['file_name']}.pth", map_location='cuda', weights_only=False)
    dataset = TensorDataset(*data)

    scaler = pickle.load(open(f"{cfg['data_path']}/{cfg['file_name']}_scaler.pkl", "rb"))
    for key1 in scaler.keys():
        for key2 in scaler[key1]:
            scaler[key1][key2] = scaler[key1][key2].to('cuda')

    model = TransformerModel(TransformerConfig(**cfg["model"]))

    loss_fn = Loss(ctrl_weight=cfg['train']["loss_weights"][0], state_weight=cfg['train']["loss_weights"][
        1], pairwise_weight=cfg['train']["loss_weights"][2], miso_method=cfg['miso_method'], env=cfg['train'][
        'env'], scaler=scaler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']["lr"], weight_decay=cfg['train']["weight_decay"])

    if cfg.get("auto_resume", 1):
        ckpt_root = f"{cfg['ENV_ROOT']}/wandb/{wandb.run.name}/checkpoints"
        if not os.path.exists(ckpt_root):
            checkpoint_epochs = []
        else:
            checkpoint_epochs = [int(f.split('.')[0]) for f in os.listdir(ckpt_root) if f.split('.')[0] != 'best']
        if len(checkpoint_epochs) > 0:
            last_checkpoint = max(checkpoint_epochs)
            last_checkpoint_path = f"{ckpt_root}/{last_checkpoint}.pth"
            print(f"Auto-resume: last checkpoint from {last_checkpoint_path}")

            checkpoint = torch.load(last_checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 1
            print(f"Auto-resume: no checkpoints found under {ckpt_root}")
    else:
        start_epoch = 1

    return model, optimizer, loss_fn, dataset, start_epoch


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    """
    Train the model for one epoch
    """
    train_loss, train_ctrl_loss, train_state_loss, train_pd_loss = 0., 0., 0., 0.
    for batch_idx, data in enumerate(dataloader):
        net_inputs, loss_inputs = handle_data(data)
        _predicted_input_trajectory = model(net_inputs)
        weighted_loss, ctrl_loss, state_loss, pd_loss = loss_fn(_predicted_input_trajectory=_predicted_input_trajectory, **loss_inputs)

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


def evaluate(model, dataloader, loss_fn):
    """
    Evaluate the model on the validation
    """
    eval_loss, eval_ctrl_loss, eval_state_loss, eval_pd_loss = 0., 0., 0., 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            net_inputs, loss_inputs = handle_data(data)
            _predicted_input_trajectory = model(net_inputs)
            weighted_loss, ctrl_loss, state_loss, pd_loss = loss_fn(_predicted_input_trajectory=_predicted_input_trajectory, **loss_inputs)

            eval_loss += weighted_loss.item()
            eval_ctrl_loss += ctrl_loss.item()
            eval_state_loss += state_loss.item()
            eval_pd_loss += pd_loss.item()

            avg_eval_loss = eval_loss / len(dataloader)
            avg_eval_ctrl_loss = eval_ctrl_loss / len(dataloader)
            avg_eval_state_loss = eval_state_loss / len(dataloader)
            avg_eval_pd_loss = eval_pd_loss / len(dataloader)
        return avg_eval_loss, avg_eval_ctrl_loss, avg_eval_state_loss, avg_eval_pd_loss


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_dataloader, val_dataloader):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def train(self, num_epochs: int, start_epoch: int = 1):
        """
        Train the model for num_epochs epochs
        """
        ckpt_root = f"{cfg['ENV_ROOT']}/wandb/{wandb.run.name}/checkpoints"
        os.makedirs(ckpt_root, exist_ok=True)

        with (tqdm(total=num_epochs - start_epoch + 1) as pbar):
            avg_train_loss, avg_val_loss = 0., 0.  # For the progress bar
            best_state_val_loss = float('inf')
            for epoch in range(start_epoch, num_epochs + 1):
                self.model.train(True)

                avg_train_loss, avg_train_ctrl_loss, avg_train_state_loss, avg_train_pd_loss, grad_norm = train_one_epoch(self.model, self.train_dataloader, self.optimizer, self.loss_fn)

                if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                    self.model.eval()
                    avg_val_loss, avg_val_ctrl_loss, avg_val_state_loss, avg_val_pd_loss = evaluate(self.model, self.val_dataloader, self.loss_fn)

                    # save model checkpoint (best, last)
                    self.save_checkpoint(self.model, self.optimizer, epoch, f"{ckpt_root}/{epoch}.pth")
                    if avg_val_state_loss < best_state_val_loss:
                        best_state_val_loss = avg_val_state_loss
                        self.save_checkpoint(self.model, self.optimizer, epoch, f"{ckpt_root}/best.pth")

                pbar.set_description(f'Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Grad Norm: {grad_norm:.2f}')
                pbar.update(1)

                # Log to wandb
                if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                    wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                               "train_ctrl_loss": avg_train_ctrl_loss, "val_ctrl_loss": avg_val_ctrl_loss,
                               "train_state_loss": avg_train_state_loss, "val_state_loss": avg_val_state_loss,
                               "train_pd_loss": avg_train_pd_loss, "val_pd_loss": avg_val_pd_loss, "epoch": epoch,
                               "lr": self.optimizer.param_groups[0]['lr'], "grad_norm": grad_norm})
                else:
                    wandb.log({"train_loss": avg_train_loss, "train_ctrl_loss": avg_train_ctrl_loss,
                               "train_state_loss": avg_train_state_loss, "train_pd_loss": avg_train_pd_loss,
                               "epoch": epoch, "lr": self.optimizer.param_groups[0]['lr'], "grad_norm": grad_norm})

    @staticmethod
    def _log_to_wandb(epoch, optimizer, avg_train_loss, avg_val_loss, avg_train_ctrl_loss, avg_val_ctrl_loss, avg_train_state_loss, avg_val_state_loss, grad_norm):
        log_dict = {"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                    "train_ctrl_loss": avg_train_ctrl_loss, "val_ctrl_loss": avg_val_ctrl_loss,
                    "train_state_loss": avg_train_state_loss, "val_state_loss": avg_val_state_loss,
                    "epoch": epoch, "lr": optimizer.param_groups[0]['lr'], "grad_norm": grad_norm}
        wandb.log(log_dict)

    @staticmethod
    def save_checkpoint(model, optimizer, epoch, checkpoint_path):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def main(wandb_kwargs: dict = None):
    with wandb.init(mode='disabled', **wandb_kwargs):
        cfg = dict(wandb.config)
        print(cfg)
        torch.manual_seed(cfg['train']['seed'])

        # Load the training objects
        model, optimizer, loss_fn, dataset, start_epoch = load_trainer_objs(cfg=cfg)

        # Prepare the dataloader for the distributed training
        train_data, val_data = prepare_dataloaders(dataset, cfg=cfg)

        # Create a Trainer object
        trainer = Trainer(model, optimizer, loss_fn, train_data, val_data)

        # Train the model
        trainer.train(cfg['train']['epochs'], start_epoch=start_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--env', type=str, default="", help='Environment to train (cartpole, reacher, nuplan)')
    parser.add_argument('--num_predictions', type=int, default=16, help='Number of predictions')
    parser.add_argument('--miso_method', type=str, default='miso-wta', help='MISO method (miso-pd, miso-mix, miso-wta, none)')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--auto_resume', type=int, default=1, help='Resume from last checkpoint')
    parser.add_argument('--run_id', type=str, default="", help='Run ID to resume')
    parser.add_argument('--sweep_id', type=str, default="", help='Sweep ID')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = OmegaConf.create()

    training_cfg = OmegaConf.create(OmegaConf.load("training/configs/training.yaml"))
    training_cfg['ENV'] = args.env
    training_cfg['seed'] = args.seed
    training_cfg['num_predictions'] = args.num_predictions
    training_cfg['miso_method'] = args.miso_method
    training_cfg = OmegaConf.to_container(training_cfg, resolve=True)[args.env]

    dataset_cfg = OmegaConf.load("training/configs/dataset.yaml")
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)[args.env]

    cfg = OmegaConf.merge(training_cfg, dataset_cfg, cfg, OmegaConf.from_cli(args.opts))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["auto_resume"] = args.auto_resume
    wandb_kwargs = {"project": f"miso-{args.env}", "entity": 'crml', "dir": f"{cfg['ENV_ROOT']}", "config": cfg}

    if args.run_id:
        runs = get_wandb_runs(wandb_kwargs['entity'], wandb_kwargs['project'])
        run_idx = runs['id'].index(args.run_id)
        run_id = runs['id'][run_idx]
        cfg = runs["config"][run_idx]
        wandb_kwargs.update({"resume": "allow", "id": run_id})

    if args.sweep_id:
        wandb.agent(sweep_id=args.sweep_id, function=lambda: main(wandb_kwargs=wandb_kwargs), **wandb_kwargs)
    else:
        wandb_kwargs.update({"config": cfg})
        main(wandb_kwargs=wandb_kwargs)
