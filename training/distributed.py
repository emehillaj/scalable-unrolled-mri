"""Process-group setup (torchrun or SLURM) and the few collectives the trainer uses.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

_RANK = 0


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _ranks_from_env():
    """(rank, world_size, local_rank) from torchrun, then SLURM (srun), else one process."""
    env = os.environ
    if "RANK" in env and "WORLD_SIZE" in env:
        rank = int(env["RANK"])
        return rank, int(env["WORLD_SIZE"]), int(env.get("LOCAL_RANK", rank))
    # unlike SLURM_NTASKS, SLURM_STEP_NUM_TASKS is set only inside srun tasks
    ntasks = int(env.get("SLURM_STEP_NUM_TASKS", "1"))
    if "SLURM_PROCID" in env and ntasks > 1:
        missing = [k for k in ("MASTER_ADDR", "MASTER_PORT") if k not in env]
        if missing:
            raise RuntimeError(
                f"SLURM multi-task run without {missing}: export MASTER_ADDR=$(hostname) "
                "MASTER_PORT=29500 before srun (see Multi-GPU training in the README)"
            )
        return int(env["SLURM_PROCID"]), ntasks, int(env.get("SLURM_LOCALID", "0"))
    return 0, 1, 0


def init_distributed(device=None) -> DistInfo:
    """Initialize the process group when launched with several processes (NCCL, or gloo on CPU)."""
    global _RANK
    rank, world_size, local_rank = _ranks_from_env()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=60),
        )
    _RANK = rank
    return DistInfo(rank, world_size, local_rank, device)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def print0(*args, **kwargs) -> None:
    """print() on rank 0 only (flushed by default)."""
    if _RANK == 0:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place sum over ranks (no-op in a single process)."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_params(module: torch.nn.Module, src: int = 0) -> None:
    """Copy all parameters and buffers from rank ``src`` to every rank."""
    if is_distributed():
        for _, tensor in sorted(module.state_dict().items()):
            dist.broadcast(tensor, src=src)


def broadcast_flag(flag: bool, device: torch.device, src: int = 0) -> bool:
    """Rank ``src``'s value of ``flag`` on every rank."""
    if not is_distributed():
        return bool(flag)
    t = torch.tensor([int(bool(flag))], device=device)
    dist.broadcast(t, src=src)
    return bool(t.item())


def barrier() -> None:
    if is_distributed():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()
