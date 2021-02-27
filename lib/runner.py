import pickle
import random
import logging
import os
from typing import Any

import cv2
import torch
from torch._C import device
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

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
        optimizer = self.cfg.get_optimizer(model.parameters())
        # scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer = self.exp.load_last_train_state(
                model, model.optimizer
            )
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg["epochs"]
        train_loader = self.get_precomp_loader("train")
        val_loader = self.get_precomp_loader("dev")
        # loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(
            starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs
        ):
            self.exp.epoch_start_callback(epoch, max_epochs)
            pbar = tqdm(train_loader)
            for idx, (images, captions, lengths, ids) in enumerate(pbar):
                model.train()
                # load to gpu
                images = images.to(self.device)
                captions = captions.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # compute the embeddings
                img_emb, cap_emb, cap_lens = model(images, captions, lengths)
                # record loss
                loss = model.loss(img_emb, cap_emb, cap_lens)
                # backward
                loss.backward()
                # gradient clipping
                if model.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), model.grad_clip)
                # optimize
                optimizer.step()

                # log
                postfix_dict = {}
                postfix_dict["lr"] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(
                    epoch, max_epochs, idx, len(train_loader), loss.item()
                )
                postfix_dict["loss"] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)

                self.iters += 1
                if self.iters % self.cfg["val_step"] == 0:
                    self.eval(model, val_loader)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer)

            # Validate
            # iters += 1
            # if iters % self.cfg['val_step'] == 0:
            #     self.eval(model)
        self.exp.train_end_callback()

    def eval(self, model: Any, val_loader: DataLoader) -> float:
        self.exp.eval_start_callback(self.cfg)
        params = self.cfg["model"]["parameters"]
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader)

        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

        start = time.time()
        if params["cross_attn"] == "t2i":
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, params, shard_size=128)
        elif params["cross_attn"] == "i2t":
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, params, shard_size=128)
        else:
            raise NotImplementedError
        end = time.time()
        print("Calculate similarity time:", end - start)

        # caption retrieval
        (r1_t, r5_t, r10_t, medr_t, meanr_t) = i2t(img_embs, cap_embs, cap_lens, sims)
        self.exp.retrieval_end_callback("i2t", r1_t, r5_t, r10_t, medr_t, meanr_t)
        # image retrieval
        (r1_i, r5_i, r10_i, medr_i, meanr_i) = t2i(img_embs, cap_embs, cap_lens, sims)
        self.exp.retrieval_end_callback("t2i", r1_i, r5_i, r10_i, medr_i, meanr_i)
        # sum of recalls to be used for early stopping
        currscore = r1_t + r5_t + r10_t + r1_i + r5_i + r10_i
        results = {
            "r1_t": r1_t,
            "r5_t": r5_t,
            "r10_t": r10_t,
            "medr_t": medr_t,
            "meanr_t": meanr_t,
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
        self.cfg["datasets"][split]["parameters"]["vocab"] = self.vocab
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
