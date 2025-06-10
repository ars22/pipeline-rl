from tapeagents.remote_environment import EnvironmentServer
from omegaconf import OmegaConf


class WebEnvironmentServer:
    n_envs: int = 8
    host: str = "0.0.0.0"
    web_env_target: str = "tapeagents.examples.rl_webagent.environment.WebEnvironment"
    exp_path: str
    headless: bool = True
    observation_format: str = "html"

    def launch(self, port: int):
        """
        Serve the web environment in TapeAgent.
        """
        env_server = EnvironmentServer(n_envs=self.n_envs, host=self.host, port=port)
        env_server.launch(OmegaConf.create({
            "_target_": self.web_env_target,
            "exp_path": self.exp_path,
            "headless": self.headless,
            "observation_format": self.observation_format,
        }))

