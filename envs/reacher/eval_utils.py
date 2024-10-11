import os
import ray
from .mppi_solver import Env, MPPISolver


@ray.remote(num_cpus=1, num_gpus=0.05)
def solve_scenario_open_loop(cfg, scenario_token):
    os.environ['EXP_TYPE'] = f"{cfg['experiment']}"
    cfg['scenario_token'] = int(scenario_token)
    solver = MPPISolver(cfg)
    while solver.solve():
        pass


@ray.remote(num_cpus=1, num_gpus=0.05)
def solve_scenario_closed_loop(cfg, scenario_token):
    os.environ['EXP_TYPE'] = f"{cfg['experiment']}"
    env = Env(cfg, scenario_token=int(scenario_token))
    env.run_simulation()
