import os
import re
import logging
import subprocess
from typing import Any, Tuple

from collections import OrderedDict

from lib.config import Config

import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (0.0001 + self.count)

    def __str__(self):
        """String representation for logging"""
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return "%.4f (%.4f)" % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line"""
        s = ""
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += "  "
            s += k + " " + str(v)
        return s

    def tb_log(self, tb_logger, prefix="", step=None):
        """Log using tensorboard"""
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


class Experiment:
    def __init__(
        self,
        exp_name: str,
        args: Any = None,
        mode: str = "train",
        exps_basedir: str = "experiments",
        tensorboard_dir: str = "tensorboard",
    ):
        self.name = exp_name
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, "models")
        self.cfg_path = os.path.join(self.exp_dirpath, "config.yaml")
        self.code_state_path = os.path.join(self.exp_dirpath, "code_state.txt")
        self.log_path = os.path.join(self.exp_dirpath, "log_{}.txt".format(mode))
        self.results_path = os.path.join(
            self.exp_dirpath, "results_{}.csv".format(exp_name)
        )
        self.tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, exp_name))
        self.cfg = None
        self.setup_exp_dir()
        self.setup_logging()

        if args is not None:
            self.log_args(args)

    def setup_exp_dir(self) -> None:
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            self.save_code_state()

    def save_code_state(self) -> None:
        state = "Git hash: {}".format(
            subprocess.run(
                ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=False
            ).stdout.decode("utf-8")
        )
        state += "\n*************\nGit diff:\n*************\n"
        state += subprocess.run(
            ["git", "diff"], stdout=subprocess.PIPE, check=False
        ).stdout.decode("utf-8")
        with open(self.code_state_path, "w") as code_state_file:
            code_state_file.write(state)

    def setup_logging(self) -> None:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(
            level=logging.DEBUG, handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger(__name__)

    def log_args(self, args: Any) -> None:
        self.logger.debug("CLI Args:\n %s", str(args))

    def set_cfg(self, cfg: Config, override: bool = False) -> None:
        assert "model_checkpoint_interval" in cfg
        self.cfg = cfg
        if not os.path.exists(self.cfg_path) or override:
            with open(self.cfg_path, "w") as cfg_file:
                cfg_file.write(str(cfg))

    def get_last_checkpoint_epoch(self) -> int:
        pattern = re.compile("model_(\\d+).pt")
        last_epoch = -1
        for ckpt_file in os.listdir(self.models_dirpath):
            result = pattern.match(ckpt_file)
            if result is not None:
                epoch = int(result.groups()[0])
                if epoch > last_epoch:
                    last_epoch = epoch

        return last_epoch

    def get_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.models_dirpath, "model_{:04d}.pt".format(epoch))

    def get_epoch_model(self, epoch: int) -> Any:
        return torch.load(self.get_checkpoint_path(epoch))["model"]

    def load_last_train_state(self, model: Any, optimizer: Any) -> Tuple[int, Any, Any]:
        epoch = self.get_last_checkpoint_epoch()
        train_state_path = self.get_checkpoint_path(epoch)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state["model"])
        optimizer.load_state_dict(train_state["optimizer"])

        return epoch, model, optimizer

    def save_train_state(self, epoch: int, model: Any, optimizer: Any) -> None:
        train_state_path = self.get_checkpoint_path(epoch)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
            },
            train_state_path,
        )

    def iter_end_callback(
        self, epoch: int, max_epochs: int, iter_nb: int, max_iter: int, loss: float
    ) -> None:
        line = "Epoch [{}/{}] - Iter [{}/{}] - Loss: {:.5f} - ".format(
            epoch, max_epochs, iter_nb, max_iter, loss
        )
        self.logger.debug(line)
        overall_iter = (epoch * max_iter) + iter_nb
        self.tensorboard_writer.add_scalar("loss/total_loss", loss, overall_iter)

    def epoch_start_callback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug(f"Epoch [{epoch}/{max_epochs}] starting.")

    def epoch_end_callback(
        self, epoch: int, max_epochs: int, model: Any, optimizer: Any
    ) -> None:
        self.logger.debug(f"Epoch [{epoch}/{max_epochs}] finished.")
        if epoch % self.cfg["model_checkpoint_interval"] == 0:
            self.save_train_state(epoch, model, optimizer)

    def retrieval_end_callback(
        self, type: str, r1: int, r5: int, r10: int, medr: int, meanr: int
    ) -> None:
        type_info = "Text to image" if (type == "t2i") else "Image to text"
        r1 = r1.detach().numpy().item()
        r5 = r5.detach().numpy().item()
        r10 = r10.detach().numpy().item()
        medr = medr.detach().numpy().item()
        meanr = meanr.detach().numpy().item()
        self.logger.info(f"{type_info}: {r1}, {r5}, {r10}, {medr}, {meanr}")


    def encode_data_start_callback(self) -> Tuple[AverageMeter, LogCollector]:
        batch_time = AverageMeter()
        val_logger = LogCollector()
        return batch_time, val_logger

    def train_start_callback(self, cfg: Config) -> None:
        self.logger.debug(f"Beginning training session. CFG used: {cfg}")

    def train_end_callback(self) -> None:
        self.logger.debug("Training session finished.")

    def eval_start_callback(self, cfg: Config) -> None:
        self.logger.debug(f"Beginning testing session. CFG used:{cfg}")

    def eval_end_callback(
        self, dataset_split, iter_evaluated: int, results: dict
    ) -> None:
        self.logger.debug(
            f"Testing session finished on model after iter {iter_evaluated}"
        )
        # self.logger.info("Results:\n %s", str(results))
        # Log tensorboard metrics
        for key in results:
            self.tensorboard_writer.add_scalar(
                "{}_metrics/{}".format(dataset_split, key),
                results[key].detach().numpy().item(),
                iter_evaluated,
            )
