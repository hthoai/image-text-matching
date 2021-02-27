# python main.py --mode train --exp_name scan_0227-0600 --cfg config/scan.yml

import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image Text Matching")
    parser.add_argument("--mode", default="train", help="Train or test?")
    parser.add_argument("--exp_name", default="debug", help="Experiment name")
    parser.add_argument("--cfg", default="config/scan.yml", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument(
        "--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU"
    )
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception(
            "If you are training, you have to set a config file using --cfg /path/to/your/config.yaml"
        )
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == "train":
        raise Exception("The `epoch` parameter should not be set when training")

    return args


def main() -> None:
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available() or args.cpu
        else torch.device("cuda")
    )
    runner = Runner(cfg, exp, device, resume=args.resume)
    if args.mode == "train":
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info("Training interrupted.")
    else:
        runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch())


if __name__ == "__main__":
    main()
