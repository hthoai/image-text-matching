from typing import Any
import torch
import torch.utils.data as data
import nltk
import numpy as np


class EvalDataset(data.Dataset):

    def __init__(self, root: str, split: str, vocab: Any):
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
            self.max_length += 2 # for <start>, <end> tokens

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
        self.images = self.images[:self.length]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        # image = torch.Tensor(self.images[img_id])
        all_images = torch.Tensor(self.images)
        tokens = self.tokens[index]
        vocab = self.vocab

        # Convert tokens to word ids.
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return all_images, target, index, img_id

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
    # data.sort(key=lambda x: len(x[1]), reverse=True)
    print(type(data), len(data))
    print(type(data[0]), len(data[0]))
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
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
