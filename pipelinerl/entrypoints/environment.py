import hydra
from omegaconf import DictConfig

from pipelinerl.utils import better_crashing


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("environment"):
        environment = hydra.utils.instantiate(cfg.environment)
        port = cfg.jobs[cfg.me.job_idx]["port"]
        environment.launch(port=port)


if __name__ == "__main__":
    hydra_entrypoint()
