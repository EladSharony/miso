import argparse
import os
import pickle

from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler


def data_preprocessing(cfg, dtype=torch.float32):
    dtype = dtype
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    shape_dict = {key: eval(value) for key, value in cfg['shapes'].items()}
    data = {key: [] for key in shape_dict.keys()}
    shape_dict.update({f'_{key}': eval(value) for key, value in cfg['shapes'].items()})
    keys = list(shape_dict.keys())

    for token in tqdm(cfg['token_list']):
        try:
            with np.load(f"{cfg['logs_path']}/{cfg['file_name']}_{token}.npz", allow_pickle=True) as d:
                [data[key].extend(d[key]) for key in data.keys()]
        except Exception as e:
            print(f"Error loading {token}: {e}")

    scaler = dict(zip(data.keys(), [StandardScaler() for _ in range(len(keys)//2)]))

    for key in tqdm(scaler.keys()):
        data[key] = np.array(data[key], dtype=np_dtype).reshape(-1, *shape_dict[key])
        data[f'_{key}'] = scaler[key].fit_transform(data[key].reshape(-1, shape_dict[key][-1])).reshape(-1, *shape_dict[key])
        data[key] = torch.from_numpy(data[key]).to(dtype)
        data[f'_{key}'] = torch.from_numpy(data[f'_{key}']).to(dtype)

    scaler = {key: {'mean_': torch.tensor(scaler[key].mean_), 'scale_': torch.tensor(scaler[key].scale_)} for key in scaler.keys()}

    return tuple(data.values()), scaler


def inverse_transform(x, scaler, key: str):
    return x * scaler[key]['scale_'] + scaler[key]['mean_']


def transform(x, scaler, key: str):
    return (x - scaler[key]['mean_']) / scaler[key]['scale_']


def main(env):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'configs/dataset.yaml'))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg[env])

    if not os.path.exists(f"{cfg['data_path']}/{cfg['file_name']}.pth"):
        with open(f"{cfg['data_path']}/train_tokens.txt", "r") as f:
            tokens = f.readlines()
            tokens = [token.replace("\n", "") for token in tokens]

        cfg['token_list'] = tokens[:cfg['num_tokens']]
        data, scaler = data_preprocessing(cfg)

        torch.save(data, f"{cfg['data_path']}/{cfg['file_name']}.pth")
        pickle.dump(scaler, open(f"{cfg['data_path']}/{cfg['file_name']}_scaler.pkl", "wb"))

        print(f"Environment {env} processed successfully: {len(data[0])} data points")
    else:
        print(f"Environment {env} already processed")


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset script')
    parser.add_argument('--env', type=str, default="", help='Environment to train')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.env)
