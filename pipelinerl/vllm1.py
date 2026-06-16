import asyncio
import logging
import shutil
import signal
from typing import Any, Protocol, runtime_checkable

import torch
import uvloop
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    build_app,
    init_app_state,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.usage.usage_lib import UsageContext
from vllm.config import ModelConfig
from vllm.platforms import current_platform
from vllm.version import __version__ as VLLM_VERSION
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from pipelinerl.torch_utils import stateless_init_process_group
from pipelinerl.trainer_messages import WeightUpdateRequest

logger = logging.getLogger(__name__)
# configure this logger individually, in order to avoid messign
# with the default vllm logger configuration
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def _translate_weight_name_for_vllm(vllm_model_name: str, source_name: str) -> tuple[str | None, str | None]:
    if vllm_model_name not in {"Qwen3_5ForConditionalGeneration", "qwen3_5"}:
        return source_name, None

    if source_name == "lm_head.weight":
        # The Qwen3.5 vLLM model ties output embeddings internally and does not
        # expose lm_head as a standalone parameter.
        return None, None

    if not source_name.startswith("model."):
        return source_name, None

    return f"language_model.{source_name}", None


def _requires_ninja_for_qwen35(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    return model_type == "qwen3_5" and current_platform.is_device_capability(90)


def _ensure_qwen35_runtime_dependencies(vllm_config: Any) -> None:
    if not _requires_ninja_for_qwen35(vllm_config):
        return
    if shutil.which("ninja") is not None:
        return
    raise RuntimeError(
        "Qwen3.5 inference on SM90 requires the `ninja` executable in the active runtime. "
        "Install `ninja` into the `prl` environment or add it to PATH before launching this model."
    )


@runtime_checkable
class LikeWorker(Protocol):
    rank: int
    local_rank: int
    device: torch.device
    model_runner: GPUModelRunner
    pg_rank: int
    model_config: ModelConfig


class WorkerExtension:
    @staticmethod
    def _resolve_dtype(value: Any) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        dtype_name = str(value).replace("torch.", "")
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported dtype for weight update: {value}")
        return mapping[dtype_name]

    def init_actor_update_group(
        self: LikeWorker,
        actor_idx: int,
        actor_ngpus: int,
        weight_update_group_init_method: str,
        weight_update_group_world_size: int,
    ):
        self.pg_rank = 1 + actor_idx * actor_ngpus + self.rank
        # log all you know
        prefix = "[INIT_ACTOR_UPDATE_GROUP]: "
        logger.info(
            prefix
            + f"Actor index: {actor_idx}, actor ngpus: {actor_ngpus}, rank: {self.rank}, pg_rank: {self.pg_rank}"
        )
        logger.info(
            prefix
            + f"Weight update group init method: {weight_update_group_init_method}, world size: {weight_update_group_world_size}"
        )
        self.model_update_group = stateless_init_process_group(
            init_method=weight_update_group_init_method,
            rank=self.pg_rank,
            world_size=weight_update_group_world_size,
            device=self.device,
        )
        logger.info(prefix + "Actor update process group initialized")

    def receive_weight_update(self: LikeWorker, request_json: str):
        request = WeightUpdateRequest.model_validate_json(request_json)
        torch.cuda.synchronize(self.device)
        logger.info("Start receiving weight update")
        model_name = (
            getattr(getattr(self.model_config, "hf_config", None), "model_type", None)
            or type(self.model_runner.model).__name__
        )
        for info in request.parameters_info:
            buffer = torch.empty(
                tuple(info.shape),
                dtype=self._resolve_dtype(info.dtype),
                device=self.device,
            )
            self.model_update_group.broadcast(buffer, src=0, stream=torch.cuda.current_stream())
            target_name, fused_part = _translate_weight_name_for_vllm(model_name, info.name)
            if target_name is None:
                continue

            try:
                loaded_params = self.model_runner.model.load_weights(weights=[(target_name, buffer)])  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load trainer parameter {info.name} into vLLM parameter {target_name}"
                ) from exc
            if len(loaded_params) != 1:
                raise ValueError(f"model {target_name} not found in model state dict")
        logger.info("Weight update received")

    def close_communicator(self: LikeWorker):
        if hasattr(self, "model_update_group") and self.model_update_group is not None:
            del self.model_update_group
            self.model_update_group = None
            logger.info("Weight update communicator closed")


class WeightUpdateManager:
    def __init__(self, args, engine: AsyncLLM, engine_client: AsyncMPClient):
        self.args = args
        self.engine = engine
        self.engine_client = engine_client

    async def input_process_groups(self):
        await self.engine_client.collective_rpc_async(
            "init_actor_update_group",
            args=(
                self.args.actor_llm_idx,
                torch.cuda.device_count(),
                self.args.weight_update_group_init_method,
                self.args.weight_update_group_world_size,
            ),
        )

    async def receive_weight_update(self, request: WeightUpdateRequest):
        logger.info("Starting weight update...")
        await self.engine_client.collective_rpc_async(
            "receive_weight_update", args=(request.model_dump_json(),)
        )
        logger.info("Weight update processed")

    async def close_communicator(self):
        await self.engine_client.collective_rpc_async("close_communicator")


async def run_server(args, **uvicorn_kwargs) -> None:
    # COPIED FROM vllm/entrypoints/openai/api_server.py, vllm version 0.6.6.post1
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if getattr(args, "reasoning_parser_plugin", "") and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    valid_tool_parsers = set(ToolParserManager.list_registered())
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parsers:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} (chose from {{ {','.join(sorted(valid_tool_parsers))} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "pipelinerl.vllm1.WorkerExtension"
    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    _ensure_qwen35_runtime_dependencies(engine_config)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        enable_log_requests=engine_args.enable_log_requests,
    )
    assert isinstance(engine.engine_core, AsyncMPClient)
    supported_tasks = await engine.get_supported_tasks()

    weight_update_manager = WeightUpdateManager(args, engine, engine.engine_core)
    if not args.disable_weight_updates:
        await weight_update_manager.input_process_groups()

    app = build_app(args, supported_tasks)

    @app.post("/receive_weight_update")
    async def _receive_weight_update(request: WeightUpdateRequest):
        logger.info("Received weight update request")
        await weight_update_manager.receive_weight_update(request)
        return {"status": "ok"}

    await init_app_state(engine, app.state, args, supported_tasks)
    try:
        shutdown_task = await serve_http(
            app,
            sock,
            enable_ssl_refresh=getattr(args, "enable_ssl_refresh", False),
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not getattr(args, "disable_uvicorn_access_log", False),
            timeout_keep_alive=60,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=getattr(args, "ssl_ciphers", None),
            h11_max_incomplete_event_size=getattr(args, "h11_max_incomplete_event_size", None),
            h11_max_header_count=getattr(args, "h11_max_header_count", None),
            **uvicorn_kwargs,
        )

        # NB: Await server shutdown only after the backend context is exited
        await shutdown_task
    finally:
        if not args.disable_weight_updates:
            await weight_update_manager.close_communicator()
        engine.shutdown()
        sock.close()


def run_llm():
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--disable-weight-updates", action="store_true", help="Whether to receive weight updates from the trainer"
    )
    parser.add_argument(
        "--actor-llm-idx",
        type=int,
    )
    parser.add_argument(
        "--weight-update-group-init-method",
        type=str,
    )
    parser.add_argument(
        "--weight-update-group-world-size",
        type=int,
    )
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
