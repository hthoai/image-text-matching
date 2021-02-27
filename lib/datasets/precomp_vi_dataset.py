from typing import Any
import numpy as np

import os
import json

import torch
import torch.utils.data as data

from utils.vocab_vi import ViVocabulary


class PrecompViDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, root: str, split: str):
        self.split = split
        self.max_length = 0
        loc = root + "/"
        self.vocab = ViVocabulary()
        with open(os.path.join(root, 'f30k_precomp_vocab_vi.json')) as f:
            d = json.load(f)
        self.vocab.word2id = d["word2id"]
        self.vocab.id2word = d["id2word"]

        # captions
        self.tokens = []
        with open(loc + "%s_caps_vi.txt" % split, "r") as f:
            for line in f:
                _caption = line.strip()
                to_id = self.vocab.sent2id(_caption)
                self.tokens.append(to_id)
                if len(to_id) > self.max_length:
                    self.max_length = len(to_id)

        # Image features
        self.images = np.load(loc + "%s_ims.npy" % self.split)
        self.length = len(self.tokens)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if split == "dev":
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        tokens = self.tokens[index]

        target = torch.Tensor(tokens)
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.Tensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(dataset, batch_size=128, shuffle=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader
