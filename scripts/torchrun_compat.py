"""
Drop-in `torchrun` that also works on Windows.

    python scripts/torchrun_compat.py --standalone --nproc_per_node=2 train.py --phase 1

Windows builds of PyTorch ship without libuv, yet torchrun's rendezvous creates
its TCPStore with the libuv default (and this torch version ignores USE_LIBUV=0
on that path), so plain `torchrun` dies with "use_libuv was requested but
PyTorch was built without libuv support". This wrapper makes the rendezvous
store use the classic TCP backend and then hands over to the real torchrun.
On Linux/macOS nothing is patched. The worker processes themselves already
default to use_libuv=0 on Windows (torch/distributed/rendezvous.py).
"""
import sys

if sys.platform == 'win32':
    import torch.distributed as dist

    class _TCPStoreWithoutLibuv(dist.TCPStore):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('use_libuv', False)
            super().__init__(*args, **kwargs)

    # dynamic_rendezvous calls dist.TCPStore(...); the c10d and static backends
    # imported the name directly, so patch those module globals as well.
    dist.TCPStore = _TCPStoreWithoutLibuv
    from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend, static_tcp_rendezvous
    c10d_rendezvous_backend.TCPStore = _TCPStoreWithoutLibuv
    static_tcp_rendezvous.TCPStore = _TCPStoreWithoutLibuv

from torch.distributed.run import main

if __name__ == '__main__':
    sys.exit(main())
