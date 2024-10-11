from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from utils import combine_logs_and_delete_temp_files


class CombineAndSaveILQRLogCallback(AbstractCallback):

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        metadata = setup.ego_controller._tracker._ilqr_solver.metadata
        metadata['scenario_token'] = setup.scenario.token

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        metadata = setup.ego_controller._tracker._ilqr_solver.metadata
        combine_logs_and_delete_temp_files(metadata)
