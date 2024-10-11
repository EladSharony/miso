import os
import hydra
import os.path
from pathlib import Path
from dataclasses import dataclass

from nuplan.planning.script.run_simulation import run_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard

from omegaconf import OmegaConf, open_dict
script_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths.
    """
    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str
    ego_controller_dir: str = None
    tracker_dir: str = None
    simulation_metric_dir: str = None


def set_environment_variables():
    config = OmegaConf.load(os.path.join(script_dir, "config.yaml"))['PATHS']
    OmegaConf.resolve(config)
    for key, value in config.items():
        os.environ[key] = str(value)


def initialize_hydra_and_compose_configuration(config, simulation_hydra_paths):
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=simulation_hydra_paths.config_path, version_base=None)

    simulation_config = OmegaConf.load(os.path.join(script_dir, "config.yaml"))
    OmegaConf.resolve(simulation_config)

    overrides = []
    for key, value in simulation_config['NUPLAN_SIMULATION_CONFIG'].items():
        if key == 'scenario_filter.scenario_tokens':
            value = config['token_list']
        if key.startswith('main_callback_metric_'):
            overrides.append(f"{value}")
            continue
        overrides.append(f"{key}={value}")
    overrides.append(f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]')

    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)

    with open(os.path.join(simulation_hydra_paths.tracker_dir, 'ilqr_tracker.yaml'), 'r') as tracker_file:
        tracker_cfg = OmegaConf.load(tracker_file)
        with open_dict(tracker_cfg):
            tracker_cfg.ilqr_solver.solver_params._target_ = 'envs.nuplan.ilqr_solver.ModifiedILQRSolverParameters'
            tracker_cfg.ilqr_solver._target_ = 'envs.nuplan.ilqr_solver.ModifiedILQRSolver'
            tracker_cfg.ilqr_solver.solver_params.metadata = config['metadata']

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.ego_controller.tracker = tracker_cfg

    # get all possible metrics
    dirs = [os.path.join(simulation_hydra_paths.simulation_metric_dir, 'low_level'),
            os.path.join(simulation_hydra_paths.simulation_metric_dir, 'high_level')]
    simulation_metrics = {'low_level': {}, 'high_level': {}}
    for dir in dirs:
        for file in os.listdir(os.path.abspath(dir)):
            if file.endswith('.yaml'):
                metric_cfg = OmegaConf.load(os.path.join(dir, file))
                with open_dict(metric_cfg):
                    metric_name = file.split('.')[0]
                    simulation_metrics[dir.split('/')[-1]][metric_name] = metric_cfg[metric_name]

    low_level_metrics_to_run = ['ego_lane_change_statistics']
    high_level_metrics_to_run = ['drivable_area_compliance_statistics', 'no_ego_at_fault_collisions_statistics',
                                 'time_to_collision_within_bound_statistics', 'speed_limit_compliance_statistics']
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        # cfg.simulation_metric = simulation_metrics
        cfg.simulation_metric = {'low_level': {k: simulation_metrics['low_level'][k] for k in low_level_metrics_to_run
                                               if k in simulation_metrics['low_level']},
                                 'high_level': {k: simulation_metrics['high_level'][k] for k in high_level_metrics_to_run
                                                if k in simulation_metrics['high_level']}}
        # Add CombineAndSaveILQRLogCallback callback to the callback list
        cfg.callback['combine_and_save_ilqr_logs'] = {'_target_': 'envs.nuplan.nuplan_callbacks.CombineAndSaveILQRLogCallback', '_convert_': 'all'}

    return cfg


def run_nuplan_simulation(config):
    set_environment_variables()
    simulation_hydra_paths = construct_simulation_hydra_paths()
    cfg = initialize_hydra_and_compose_configuration(config, simulation_hydra_paths)

    # Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
    run_simulation(cfg)
    # Get nuBoard simulation file for visualization later on
    simulation_file = [str(file) for file in Path(cfg.output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

    return simulation_file


def lunch_nuboard(simulation_file):
    # Location of paths with all nuBoard configs
    nuboard_hydra_paths = construct_nuboard_hydra_paths()

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=nuboard_hydra_paths.config_path)

    # Compose the configuration
    cfg = hydra.compose(config_name=nuboard_hydra_paths.config_name, overrides=[
        'scenario_builder=nuplan',
        # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={simulation_file}',
        # nuboard file path, if left empty the user can open the file inside nuBoard
        f'hydra.searchpath=[{nuboard_hydra_paths.common_dir}, {nuboard_hydra_paths.experiment_dir}]',
        f'port_number={os.getenv("PORT")}',
    ])

    # Run nuBoard
    main_nuboard(cfg)


def construct_simulation_hydra_paths():
    """
    Specifies relative paths to simulation configs
    :return: Hydra config path.
    """
    config_name = 'default_simulation'

    current_file_parent = Path(__file__).parent
    base_config_path = Path(os.environ['BASE_CONFIG_PATH']).resolve()

    relative_config_path = os.path.relpath(base_config_path, current_file_parent)

    common_dir = f"file://{base_config_path / 'config/common'}"
    config_path = os.path.join(relative_config_path, 'config', 'simulation')
    experiment_dir = f"file://{base_config_path / 'experiments'}"
    ego_controller_dir = (base_config_path / 'config/simulation/ego_controller').resolve()
    tracker_dir = (base_config_path / 'config/simulation/ego_controller/tracker').resolve()
    simulation_metric_dir = (base_config_path / 'config/common/simulation_metric').resolve()

    return HydraConfigPaths(common_dir, config_name, str(config_path),
                            experiment_dir, str(ego_controller_dir),
                            str(tracker_dir), str(simulation_metric_dir))


def construct_nuboard_hydra_paths() -> HydraConfigPaths:
    """
    Specifies relative paths to nuBoard configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    set_environment_variables()
    current_file_parent = Path(__file__).parent
    target_path = Path(os.path.join(os.getenv('NUPLAN_DEVKIT_ROOT'), 'nuplan/planning/script')).resolve()
    base_config_path = os.path.relpath(target_path, current_file_parent)

    common_dir = "file://" + os.path.join(base_config_path, 'config', 'common')
    config_name = 'default_nuboard'
    config_path = os.path.join(base_config_path, 'config/nuboard')
    experiment_dir = "file://" + os.path.join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)
