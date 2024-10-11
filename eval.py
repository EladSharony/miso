import os
import sys

from tqdm import tqdm
import ray
from omegaconf import OmegaConf
from utils import get_wandb_runs
import logging

from torch.cuda import device_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_eval(cfg, tokens, env, exp):
    if env == 'reacher':
        if exp == 'open_loop':
            from envs.reacher.eval_utils import solve_scenario_open_loop as solve_scenario
        elif exp == 'closed_loop':
            from envs.reacher.eval_utils import solve_scenario_closed_loop as solve_scenario

    elif env == 'nuplan':
        if exp == 'open_loop':
            from envs.nuplan.eval_utils import solve_scenario
        elif exp == 'closed_loop':
            from envs.nuplan.eval_utils import run_closed_loop_eval as solve_scenario

    elif env == 'cartpole':
        if exp == 'open_loop':
            from envs.cartpole.eval_utils import solve_scenario_open_loop as solve_scenario
        elif exp == 'closed_loop':
            from envs.cartpole.eval_utils import solve_scenario_closed_loop as solve_scenario

    else:
        raise ValueError("Invalid environment")

    num_cpus = int(0.7 * os.cpu_count())
    num_gpus = device_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)  # , local_mode=True)

    try:
        if env in ['reacher', 'cartpole'] or (env == 'nuplan' and exp == 'open_loop'):
            # Parallel processing using Ray
            futures = [solve_scenario.remote(cfg, token) for token in tokens]
            progress_bar = tqdm(total=len(futures), desc="Evaluation Scenarios")
            remaining_futures = futures
            while remaining_futures:
                done, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=30)
                if done:
                    for obj_ref in done:
                        try:
                            ray.get(obj_ref)
                        except Exception as e:
                            logger.error(f"Error retrieving result for {obj_ref}: {e}")
                    progress_bar.update(len(done))

            progress_bar.close()
        else:
            solve_scenario(cfg, tokens)  # nuplan closed loop already takes care of parallel processing
    finally:
        ray.shutdown()


def main(env, exp, method, eval_set):
    os.environ['EXP_TYPE'] = f'{exp}'

    config = OmegaConf.load(f"envs/{env}/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    config = dict(config)

    if method in ['NN', 'NN_ensemble', 'NN_perturbation']:
        runs_df = get_wandb_runs(wandb_entity='crml', wandb_project=f"miso-{env}")

    with open(f'envs/{env}/data/{eval_set}_tokens.txt', 'r') as f:
        tokens = f.readlines()
        tokens = [token.strip() for token in tokens]

    tokens = tokens[:2]

    cfg = {'experiment': exp, 'method': method, 'log_dir': config['PATHS']['LOG_DIR']}
    cfg.update(config[method])
    cfg.update(config['open_loop_data'])
    cfg['optimizer_mode'] = config['optimizer_mode']

    if method in ['warm_start', 'oracle']:
        print(f'{method} ({cfg["optimizer_mode"]})')
        run_eval(cfg, tokens, env, exp)

    if method == 'warm_start_perturbation':
        num_perturbations_list = config[method]['num_perturbations']
        print(f'Warm-start Perturbations  ({cfg["optimizer_mode"]}): ', num_perturbations_list)
        with tqdm(num_perturbations_list, desc="Warm-start Perturbations", position=1, leave=False) as pbar_perturb:
            for num_perturbations in pbar_perturb:
                cfg['num_perturbations'] = num_perturbations
                run_eval(cfg, tokens, env, exp)

    if method == 'NN_perturbation':
        num_perturbations_list = config[method]['num_perturbations']
        print(f'NN Perturbations ({cfg["optimizer_mode"]}): ', num_perturbations_list)
        with tqdm(num_perturbations_list, desc="NN Perturbations", position=1, leave=False) as pbar_nn_perturb:
            for num_perturbations in pbar_nn_perturb:
                cfg['num_perturbations'] = num_perturbations
                cfg['run_name'] = config[method]['run_name']
                run_config = runs_df[runs_df['name'] == cfg['run_name']]['config'].values[0]
                cfg['model_type'] = run_config['model_type']
                cfg['model'] = run_config['model']
                run_eval(cfg, tokens, env, exp)

    if method == 'NN_ensemble':
        ensemble_size_list = config[method]['ensemble_size']
        print(f'NN Ensemble ({cfg["optimizer_mode"]}): ', ensemble_size_list)
        with tqdm(ensemble_size_list, desc="NN Ensemble", position=1, leave=False) as pbar_ensemble:
            for ensemble_size in pbar_ensemble:
                cfg['run_name'] = config[method]['run_name'][:ensemble_size]
                run_config = runs_df[runs_df['name'] == cfg['run_name'][0]]['config'].values[0]
                cfg['model_type'] = run_config['model_type']
                cfg['model'] = run_config['model']
                run_eval(cfg, tokens, env, exp)

    if method == 'NN':
        run_names = config[method]['run_name']
        print(f'NN ({cfg["optimizer_mode"]}): ', run_names)
        with tqdm(run_names, desc="Runs", position=1, leave=False) as pbar_runs:
            for run_name in pbar_runs:
                cfg['run_name'] = run_name
                run_config = runs_df[runs_df['name'] == cfg['run_name']]['config'].values[0]
                cfg['model_type'] = run_config['model_type']
                cfg['model'] = run_config['model']
                run_eval(cfg, tokens, env, exp)


if __name__ == '__main__':
    env = 'nuplan'  # 'cartpole', 'reacher', 'nuplan'
    eval_set = 'train'  # 'train', 'test'
    exp = 'closed_loop'  # 'open_loop', 'closed_loop'
    method = 'warm_start'  # 'warm_start', 'warm_start_perturbation', 'oracle', 'NN', 'NN_ensemble', 'NN_perturbation'

    # Modify sys.path to include the necessary directories
    base_dir = os.path.abspath(os.path.dirname(__file__))
    env_dir = os.path.join(base_dir, 'envs', env)

    sys.path.append(os.path.join(base_dir, 'envs'))
    sys.path.append(env_dir)

    main(env, exp, method, eval_set)
