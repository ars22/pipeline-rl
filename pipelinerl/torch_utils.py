import logging
from datetime import timedelta
from typing import Any, Optional, Union
from urllib.parse import urlparse

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    ProcessGroupNCCL,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


def stateless_init_process_group(init_method, rank, world_size, device):
    parsed = urlparse(init_method)
    master_address = parsed.hostname or "localhost"
    master_port = parsed.port or 9000
    process_group = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    return PyNcclCommunicator(process_group, device=device)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_extra_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    if pg_options is None and str(backend) == "nccl":
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False

    logger.info(
        "[%s] Creating process group rank=%s world_size=%s backend=%s",
        group_name,
        rank,
        world_size,
        backend,
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    logger.info("[%s] Process group created", group_name)
    return pg
