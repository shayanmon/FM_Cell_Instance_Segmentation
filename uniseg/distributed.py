# ============================================================
# Distributed Training Setup (DDP)
# ============================================================
# Works with torchrun (local workstation or HPC) and with plain `srun`
# (one task per GPU). Backend selection:
#   - NCCL on Linux when every local rank has its own GPU (the HPC case)
#   - gloo otherwise: Windows has no NCCL, and gloo is also the only backend
#     that allows several ranks to share one GPU.
import os
import sys
import datetime
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    active: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    backend: str = 'none'
    device: torch.device = torch.device('cpu')

    @property
    def is_main(self):
        return self.rank == 0


def _read_launcher_env():
    """Return (rank, local_rank, world_size, local_world_size) or None if not launched distributed."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:   # torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', world_size))
        return rank, local_rank, world_size, local_world_size
    if 'SLURM_PROCID' in os.environ and int(os.environ.get('SLURM_NTASKS', 1)) > 1:   # srun
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        # e.g. "2" or "2(x3)" for 2 tasks on each of 3 nodes
        local_world_size = int(os.environ.get('SLURM_NTASKS_PER_NODE', str(world_size)).split('(')[0])
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        return rank, local_rank, world_size, local_world_size
    return None


def init_distributed(backend='auto', timeout_minutes=60):
    """Initialise the process group (if launched distributed) and pick this rank's device."""
    launch = _read_launcher_env()
    n_gpus = torch.cuda.device_count()

    if launch is None:
        device = torch.device('cuda:0' if n_gpus else 'cpu')
        if n_gpus:
            torch.cuda.set_device(device)
        return DistEnv(device=device)

    rank, local_rank, world_size, local_world_size = launch
    if n_gpus:
        # More local ranks than GPUs -> ranks share GPUs round-robin (gloo only)
        device = torch.device(f'cuda:{local_rank % n_gpus}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if backend == 'auto':
        gpu_per_rank = n_gpus > 0 and local_world_size <= n_gpus
        use_nccl = dist.is_nccl_available() and sys.platform != 'win32' and gpu_per_rank
        backend = 'nccl' if use_nccl else 'gloo'

    timeout = datetime.timedelta(minutes=timeout_minutes)
    if sys.platform == 'win32':
        # Windows PyTorch has no libuv, but TCPStore clients still default to it.
        # Build the store explicitly. Under torchrun the agent already hosts it
        # at MASTER_ADDR:MASTER_PORT, so every worker is a client.
        agent_hosts_store = os.environ.get('TORCHELASTIC_USE_AGENT_STORE', '').lower() == 'true'
        store = dist.TCPStore(
            host_name=os.environ['MASTER_ADDR'],
            port=int(os.environ['MASTER_PORT']),
            world_size=world_size,
            is_master=(rank == 0 and not agent_hosts_store),
            timeout=timeout,
            use_libuv=False,
        )
        dist.init_process_group(backend=backend, store=store, rank=rank,
                                world_size=world_size, timeout=timeout)
    else:
        dist.init_process_group(backend=backend, init_method='env://', rank=rank,
                                world_size=world_size, timeout=timeout)
    return DistEnv(active=True, rank=rank, local_rank=local_rank, world_size=world_size,
                   local_world_size=local_world_size, backend=backend, device=device)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if rank 0 or not in DDP mode."""
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _gloo_needs_host_copy(tensor):
    # gloo's CUDA all_reduce crashes (access violation) on Windows builds; gloo
    # stages CUDA tensors through host memory anyway, so reduce on the CPU.
    return tensor.is_cuda and dist.get_backend() == 'gloo'


def all_reduce_sum(tensor):
    """In-place SUM across ranks (no-op when single-process)."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    if _gloo_needs_host_copy(tensor):
        cpu = tensor.detach().cpu()
        dist.all_reduce(cpu, op=dist.ReduceOp.SUM)
        tensor.copy_(cpu)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gloo_host_allreduce_hook(process_group, bucket):
    """DDP communication hook: average a gradient bucket through host memory.

    Equivalent to DDP's default all-reduce-then-divide hook, but avoids gloo's
    CUDA all_reduce, which segfaults on Windows. Register it with
    `ddp_model.register_comm_hook(None, gloo_host_allreduce_hook)` when the
    backend is gloo and the model lives on a GPU.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    grad = bucket.buffer()
    cpu = grad.detach().cpu()           # blocking D2H copy: gradients are ready
    dist.all_reduce(cpu, op=dist.ReduceOp.SUM, group=group)
    cpu.div_(dist.get_world_size(group))
    grad.copy_(cpu)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


def broadcast_object(obj, src=0):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def unwrap(model):
    """Return the underlying module of a DDP- and/or torch.compile-wrapped model."""
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def no_sync_if(model, condition):
    """DDP `no_sync()` context for gradient-accumulation micro-steps."""
    if condition and hasattr(model, 'no_sync'):
        return model.no_sync()
    return nullcontext()
