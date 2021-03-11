from typing import Any
import torch
import torch.utils.data as data
import nltk
import numpy as np


class EvalDataset(data.Dataset):
    """Read captions only."""

    def __init__(self, root: str, split: str, vocab: Any, num_imgs: int):
        self.split = split
        self.vocab = vocab
        self.max_length = 0
        loc = root + "/"

        # Tokens
        self.tokens = []
        with open(loc + "%s_caps.txt" % split, "rb") as f:
            for line in f:
                # tokenize
                _tokens = nltk.tokenize.word_tokenize(str(line.strip()).lower())
                self.tokens.append(_tokens)
                if len(_tokens) > self.max_length:
                    self.max_length = len(_tokens)
            self.max_length += 2  # for <start>, <end> tokens

        # Image features
        # self.images = np.load(loc + "%s_ims.npy" % self.split)
        self.length = len(self.tokens)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if num_imgs != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if split == "dev":
            self.length = 5000
        # self.images = self.images[:self.length]

    def __getitem__(self, index: int):
        # handle the image redundancy
        img_id = index // self.im_div
        # image = torch.Tensor(self.images[img_id])
        tokens = self.tokens[index]
        vocab = self.vocab

        # Convert tokens to word ids.
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return target, index, img_id

    def __len__(self):
        return self.length


def eval_collate_fn(data):
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
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.LongTensor(lengths)

    return targets, lengths, ids

