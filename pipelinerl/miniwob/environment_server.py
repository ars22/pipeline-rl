from tapeagents.remote_environment import EnvironmentServer
from omegaconf import OmegaConf


class WebEnvironmentServer:

    def __init__(self,
        n_envs: int,
        host: str,
        web_env_target: str,
        exp_path: str,
        headless: bool = True,
        observation_format: str = "html"
    ):
        self.n_envs = n_envs
        self.host = host
        self.web_env_target = web_env_target
        self.exp_path = exp_path
        self.headless = headless
        self.observation_format = observation_format

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

