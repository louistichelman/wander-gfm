"""Main entry point for Wander."""

import os
import signal
import socket
import faulthandler
from datetime import timedelta
import torch
import torch.distributed as dist
import wandb
from config.arguments import get_argument_parser, validate_args
from config.constants import WANDB_API_KEY


from experiment import Experiment


def _install_stackdump_handler():
    """Register a SIGUSR1 handler that dumps all thread stacks to a file."""
    if not hasattr(signal, "SIGUSR1"):
        return
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    dump_path = os.path.join(
        logs_dir, f"stackdump_{socket.gethostname()}_{os.getpid()}.txt"
    )
    dump_file = open(dump_path, "w")
    faulthandler.register(signal.SIGUSR1, file=dump_file, all_threads=True, chain=True)
    print(f"[stackdump] SIGUSR1 handler registered -> {dump_path}", flush=True)


_install_stackdump_handler()


def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU fallback."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
    return local_rank, dist.get_world_size()


def main():
    """Main entry point."""
    rank, world_size = setup_distributed()

    parser = get_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
    if rank == 0 and wandb_mode != "disabled" and not wandb_disabled:
        wandb.login(key=WANDB_API_KEY)
    
    experiment = Experiment(args, rank=rank, world_size=world_size)
    results = experiment.run_many_seeds()
    
    if dist.is_initialized():
        dist.destroy_process_group()

    return results


if __name__ == "__main__":
    main()