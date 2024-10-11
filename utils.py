import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from training.model import TransformerModel, TransformerConfig

import wandb

dataset_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'training/configs/dataset.yaml'))
OmegaConf.resolve(dataset_config)


class InitPredictor:
    def __init__(self, metadata):
        self.model, self.scaler = self.load_model_and_scaler(metadata)

        if isinstance(self.model, list):
            model_cfg = self.model[0].cfg
        else:
            model_cfg = self.model.cfg

        self.src_len = model_cfg.src_len
        self.in_traj_seq_len = model_cfg.src_len
        self.ref_traj_seq_len = model_cfg.src_len

    def __call__(self, reference_trajectory: torch.Tensor, warm_start_iterate):
        # Add batch dimension
        reference_trajectory = reference_trajectory[None, ...]
        if reference_trajectory.ndim == 2:
            reference_trajectory = reference_trajectory[None, ...]
        warm_start_state_trajectory = warm_start_iterate.state_trajectory[None, ...]
        warm_start_input_trajectory = warm_start_iterate.input_trajectory[None, ...]

        if isinstance(reference_trajectory, np.ndarray):
            reference_trajectory = torch.from_numpy(reference_trajectory)
            warm_start_state_trajectory = torch.from_numpy(warm_start_state_trajectory)
            warm_start_input_trajectory = torch.from_numpy(warm_start_input_trajectory)

        # if the horizon is less than the default horizon, pad
        reference_trajectory = F.pad(reference_trajectory, (0, 0, 0, warm_start_state_trajectory.shape[0] - reference_trajectory.shape[0]),
                                     mode="replicate")

        # Standardize the inputs
        _reference_trajectory = self.transform(reference_trajectory,
                                               key='reference_trajectory').to(torch.float32)
        _warm_start_state_trajectory = self.transform(warm_start_state_trajectory,
                                                      key='warm_start_state_trajectory').to(torch.float32)
        _warm_start_input_trajectory = self.transform(warm_start_input_trajectory,
                                                      key='warm_start_input_trajectory').to(torch.float32)

        _tracking_error = (_reference_trajectory - _warm_start_state_trajectory)[:, :self.src_len, :]
        net_inputs = torch.cat((_tracking_error, _warm_start_input_trajectory), dim=-1)

        with torch.no_grad():
            if isinstance(self.model, list):
                _predicted_input_trajectory = []
                for model in self.model:
                    _predicted_input_trajectory.append(model(net_inputs))
                _predicted_input_trajectory = torch.cat(_predicted_input_trajectory, dim=-2)
            else:
                _predicted_input_trajectory = self.model(net_inputs)

        _predicted_input_trajectory = _predicted_input_trajectory.squeeze(0).cpu()
        predicted_input_trajectory = self.inverse_transform(_predicted_input_trajectory, key='input_trajectory')

        if predicted_input_trajectory.ndim == 2:
            predicted_input_trajectory = predicted_input_trajectory[:, None, :]

        return predicted_input_trajectory

    @staticmethod
    def load_model_and_scaler(metadata):
        # TODO: add option in the config (currently the mpc,env, and model share the same device)
        device = metadata.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = metadata['model']

        if metadata['method'] == 'NN_ensemble':
            model = []
            for run_name in metadata['run_name']:
                path_to_model = f"{metadata['wandb_dir']}/{run_name}/checkpoints/best.pth"
                checkpoint = torch.load(path_to_model, map_location='cpu')  # when running ray local_mode=True, map_location='cpu' is needed
                model_ = TransformerModel(TransformerConfig(**model_cfg))

                model_.load_state_dict(checkpoint['model'])
                model_.to(device)
                model_.eval()
                model.append(model_)

        else:
            path_to_model = f"{metadata['wandb_dir']}/{metadata['run_name']}/checkpoints/best.pth"
            checkpoint = torch.load(path_to_model, map_location='cpu')  # when running ray local_mode=True, map_location='cpu' is needed
            model = TransformerModel(TransformerConfig(**model_cfg))

            model.load_state_dict(checkpoint['model'])
            model.to(device)
            model.eval()

        scaler = pickle.load(open(f"{metadata['data_dir']}/open_loop_oracle_scaler.pkl", "rb"))
        return model, scaler

    def inverse_transform(self, x, key: str):
        return x * self.scaler[key]['scale_'] + self.scaler[key]['mean_']

    def transform(self, x, key: str):
        if key == 'reference_trajectory' and key not in self.scaler:
            # TODO: Fix this hack - maybe by storing 'goal_state' also as 'reference_trajectory'
            key = 'goal_state'
        return (x - self.scaler[key]['mean_']) / self.scaler[key]['scale_']


def repeat_and_perturb(input_trajectory, K, noise_scale):
    if input_trajectory.ndim == 2:
        input_trajectory = input_trajectory[:, None, :]    # [T, 1, D] -> [T, K, D]
    if isinstance(input_trajectory, torch.Tensor):
        input_trajectories = input_trajectory.repeat(1, K, 1)
        input_trajectories += torch.randn_like(input_trajectories) * noise_scale
    elif isinstance(input_trajectory, np.ndarray):
        input_trajectories = np.tile(input_trajectory, (1, K, 1))
        input_trajectories += np.random.normal(0, noise_scale, input_trajectories.shape)
    return input_trajectories


def combine_logs_and_delete_temp_files(metadata):
    filename = f"{metadata['experiment']}_{metadata['method']}"
    logs_path = os.path.join(metadata['log_dir'], filename)

    # Get all npz files in the log directory that are related to the current experiment
    temp_filename = f"{metadata['experiment']}_{metadata['method']}"
    if metadata['method'] == 'NN':
        temp_filename = f"{temp_filename}_{metadata['run_name']}"
    elif metadata['method'] == 'NN_ensemble':
        temp_filename = f"{temp_filename}_{len(metadata['run_name'])}"
    elif metadata['method'] in ['NN_perturbation', 'warm_start_perturbation']:
        temp_filename = f"{temp_filename}_{metadata['num_perturbations']}"
    temp_filename = f"{temp_filename}_{metadata['scenario_token']}_"

    temp_files = []
    for temp_file in os.listdir(logs_path):
        if temp_filename in temp_file and temp_file.endswith('_temp.npz'):
            temp_files.append(temp_file)

    # Sort the files based on the iteration number
    temp_files.sort(key=lambda x: int(x.split('_')[-2]))

    combined_files_dict = {}

    for temp_file in temp_files:
        # Split the filename on underscore
        parts = temp_file.split('_')
        filename = '_'.join(parts[:-2])  # Exclude the timestamp at the end
        temp_file_path = os.path.join(logs_path, temp_file)
        # Load the npz file
        with np.load(temp_file_path, allow_pickle=True) as data:
            # Initialize a dictionary for the method if it doesn't exist
            if filename not in combined_files_dict:
                combined_files_dict[filename] = {key: [] for key in data.files}
            # Append the data for this call to the corresponding list in the method's dictionary
            for key in data.files:
                array = data[key]
                combined_files_dict[filename][key].append(array)

    # Save combined data for each method to separate files
    for filename, data_dict in combined_files_dict.items():
        output_file = os.path.join(logs_path, f"{filename}.npz")
        # Stack the lists of arrays along a new dimension (axis=0) for each key
        combined_data = {key: np.stack(data_list, axis=0) for key, data_list in data_dict.items()}
        # Save the combined data to a npz file
        np.savez_compressed(output_file, **combined_data)

    # Delete the temporary files
    for temp_file in temp_files:
        os.remove(os.path.join(logs_path, temp_file))


def save(metadata, data):
    filename = f"{metadata['experiment']}_{metadata['method']}"
    logs_path = os.path.join(metadata['log_dir'], filename)

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if metadata['method'] == 'NN':
        filename = f"{filename}_{metadata['run_name']}"

    if metadata['method'] in ['NN_perturbation', 'warm_start_perturbation']:
        filename = f"{filename}_{metadata['num_perturbations']}"

    if metadata['method'] == 'NN_ensemble':
        filename = f"{filename}_{len(metadata['run_name'])}"

    file_path = os.path.join(logs_path, f"{filename}_{metadata['scenario_token']}")

    if data.keys() == {'args', 'kwargs'}:
        file_path = f"{file_path}_solver_init.npz"
        np.savez_compressed(file_path, **data)
    else:
        file_path = f"{file_path}_{int(metadata['iteration'])}_temp.npz"
        np.savez_compressed(file_path, **data)


def get_wandb_runs(wandb_entity: str, wandb_project: str) -> dict:
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    summary_list, config_list, name_list, id_list = [], [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)
        id_list.append(run.id)
    runs_df = {"config": config_list, "name": name_list, "id": id_list}
    return runs_df
