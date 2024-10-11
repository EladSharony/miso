import os
import ray

from .ilqr_solver import ModifiedILQRSolver
from .nuplan_utils import run_nuplan_simulation


@ray.remote(num_cpus=1, num_gpus=0.05)
def solve_scenario(metadata, scenario_token):
    os.environ['EXP_TYPE'] = f"{metadata['experiment']}"
    metadata['scenario_token'] = scenario_token
    solver = ModifiedILQRSolver(metadata)
    while solver.solve():
        pass


def run_closed_loop_eval(metadata, tokens):
    run_nuplan_simulation({'token_list': tokens, 'metadata': metadata})
    ray.shutdown()

