import hydra
from omegaconf import DictConfig

from pipelinerl.utils import better_crashing


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):    
    with better_crashing("environment"):        
        genrm_urls_str = cfg.me.get("llm_urls", None)
        genrm_urls = genrm_urls_str.split("+") if genrm_urls_str else []
        if genrm_urls:
            print("Using GenRMMathEnvironment")
            environment = hydra.utils.instantiate(cfg.environment)
            this_job,  = [job for job in cfg.jobs if job["idx"] == cfg.me.job_idx]
            port = this_job["port"]
            environment.launch(port=port, debug_mode=bool(cfg.debug.mode), genrm_urls=genrm_urls)
        else:
            print("Using standard MathEnvironment")
            environment = hydra.utils.instantiate(cfg.environment)
            this_job,  = [job for job in cfg.jobs if job["idx"] == cfg.me.job_idx]
            port = this_job["port"]
            environment.launch(port=port, debug_mode=bool(cfg.debug.mode))


if __name__ == "__main__":
    hydra_entrypoint()
