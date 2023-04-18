import matplotlib.pyplot as plt
import torch
from typing import Iterable, Union
from torch import Tensor


def grayscale(images: Union[Tensor, Iterable[Tensor]], image_bits: int, titles: Union[str, Iterable[str]] = None, display_bits: int = 8):
    if isinstance(images, Tensor):
        images = [images]
    if isinstance(titles, str):
        titles = [titles]
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(wspace=0.1)
    for i, img in enumerate(images):
        ax = fig.add_subplot(1, len(images), i + 1)
        if image_bits > display_bits:
            img = img.to(torch.float32)
            img = torch.round(img * (2 ** display_bits - 1) / (2 ** image_bits - 1))
        ax.imshow(img, cmap='gray')
        if titles:
            ax.set_title(titles[i])
        ax.axis('off')
    plt.show()
