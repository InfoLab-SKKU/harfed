from datasets import disable_progress_bar
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
from pathlib import Path
import flwr as fl

from src.modules.attacks.attack import AttackFactory
from src.modules.client import ClientFactory
from src.modules.partitioner import PartitionFactory
from src.modules.plot import smooth_plot
from src.modules.strategy import StrategyFactory

@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: DictConfig):
    # Print the current configuration for reference
    print(OmegaConf.to_yaml(cfg))

    # Define the path for saving results, derived from Hydra's output directory
    save_path = Path(HydraConfig.get().runtime.output_dir)

    fds = PartitionFactory.get_fds(cfg)
    centrelized_testset = fds.load_split("test")

    #attack = AttackFactory.create_attack(cfg)


    #exit()
    # Start the federated learning simulation
    history = fl.simulation.start_simulation(
        client_fn=ClientFactory.get_client_fn(fds, cfg),
        num_clients=cfg.client.count,
        client_resources=cfg.client.resources,
        config=fl.server.ServerConfig(num_rounds=cfg.strategy.num_rounds),
        strategy=StrategyFactory.get_strategy(centrelized_testset, cfg),
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # Disable progress bars on virtual clients
        },
    )

    # Plot the smoothed history of the simulation results
    smooth_plot(
        data=history,
        title=f"{cfg.dataset.name.split('/')[-1]} - {cfg.dataset.partitioner.name.split('Partitioner')[0]} - {cfg.client.count} clients with 10 per round",
        path=save_path,
        smoothing_window=cfg.plot.smoothing_window
    )

if __name__ == "__main__":
    # Entry point for the script
    main()
