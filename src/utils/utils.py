import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def try_remove_old_ckpt(output_dir, topk=5):
    if topk <= 0:
        return
    p = Path(output_dir) / "ckpt"
    ckpts = sorted(
        p.glob("step-*"), key=lambda x: float(x.name.split("-")[-1]), reverse=True
    )
    if len(ckpts) > topk:
        shutil.rmtree(ckpts[-1])
        logger.info(f"remove old ckpt: {ckpts[-1]}")


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.output_dir)
    elif args.writer_type == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=args.output_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer
