import random
import logging
import os

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn

from utils.vocab import deserialize_vocab
from lib.datasets.precomp_dataset import collate_fn
from lib.evaluation import *
from lib.config import Config
from lib.experiment import Experiment


class Runner:
    def __init__(
        self,
        cfg: Config,
        exp: Experiment,
        device: torch.device,
        resume: bool = False,
        deterministic: bool = False,
    ):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.logger = logging.getLogger(__name__)
        self.iters = 0

        # Fix seeds
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])

        # Load Vocabulary Wrapper
        self.vocab = deserialize_vocab(
            os.path.join("vocab", self.cfg.get_vocab_path() + ".json")
        )
        self.vocab_size = len(self.vocab)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self) -> None:
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1

        model = self.cfg.get_model(self.vocab_size)
        model = model.to(self.device)
        criterion = self.cfg.get_criterion()
        optimizer = self.cfg.get_optimizer(model.parameters())
        # scheduler = self.cfg.get_lr_scheduler(optimizer)

        if self.resume:
            last_epoch, model, optimizer = self.exp.load_last_train_state(
                model, model.optimizer
            )
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg["epochs"]

        train_loader = self.get_precomp_loader(split="train", batch_size=128)
        val_loader = self.get_precomp_loader(split="dev", batch_size=1)

        for epoch in trange(
            starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs
        ):
            self.exp.epoch_start_callback(epoch, max_epochs)
            pbar = tqdm(train_loader)
            model.train()

            for idx, (images, captions, lengths, ids) in enumerate(pbar):
                # Load to GPU
                images = images.to(self.device)
                captions = captions.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Compute similarity
                scores = model(images, captions, lengths)

                # Record loss
                loss = criterion(scores)
                # Backward
                loss.backward()
                # Gradient clipping
                if model.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), model.grad_clip)

                # Optimize
                optimizer.step()
                # Scheduler step (iteration based)
                # scheduler.step()

                # Log to progressing bar
                postfix_dict = {}
                postfix_dict["lr"] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(
                    epoch, max_epochs, idx, len(train_loader), loss.item()
                )
                postfix_dict["loss"] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)

                self.iters += 1
                if self.iters % self.cfg["val_step"] == 0:
                    model.eval()
                    self.eval(model, val_loader, self.exp)
                    model.train()
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer)

        self.exp.train_end_callback()

    def eval(self,
             model: nn.Module,
             val_loader: DataLoader,
             exp: Experiment) -> torch.Tensor:

        self.exp.eval_start_callback(self.cfg)
        model.eval()

        pbar = tqdm(val_loader)

        n_cap = None
        n_img = None
        batch_size = None

        score_matrix = np.zeros(n_img, n_cap)

        for idx, (images, captions, cap_lens) in enumerate(pbar):
            # load to gpu
            images = images.to(self.device)
            captions = captions.to(self.device)
            # compute the embeddings
            cap_id_start, cap_id_end = idx * batch_size, min((idx+1) * batch_size - 1, n_cap) 
            score_matrix[cap_id_start:cap_id_end] = model(images, captions, cap_lens)

        # image retrieval
        (r1_i, r5_i, r10_i, medr_i, meanr_i) = ComputeMetric(score_matrix)
        self.exp.retrieval_end_callback("t2i", r1_i, r5_i, r10_i, medr_i, meanr_i)
        # sum of recalls to be used for early stopping
        currscore = r1_i + r5_i + r10_i
        results = {
            "r1_i": r1_i,
            "r5_i": r5_i,
            "r10_i": r10_i,
            "medr_i": medr_i,
            "meanr_i": meanr_i,
        }
        self.exp.eval_end_callback(val_loader.dataset.split, self.iters, results)
        return currscore

    def get_precomp_loader(
        self, split: str, batch_size: int = 128, shuffle: bool = True
    ) -> DataLoader:
        """Returns torch.utils.data.DataLoader for custom coco dataset."""
        # self.cfg["datasets"][split]["parameters"]["vocab"] = self.vocab
        dataset = self.cfg.get_dataset(split)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=self._worker_init_fn_,
        )
        return data_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
