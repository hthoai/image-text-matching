import numpy as np

from torch import Tensor


def read_all_images(root: str, split: str = "dev") -> Tensor:
    """Read all images for validation/testing."""
    images = np.load(f"{root}/{split}_ims.npy")
    if split == "dev":
        images = images[:1000]
    return Tensor(images)
