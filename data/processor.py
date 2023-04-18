import torch
from torch import Tensor
from typing import Union, List, Tuple
from torchvision import transforms


def random_augmentation(x: Union[Tensor, List[Tensor], Tuple[Tensor]]):
    """
    random augmentation composed of random rotation(0, 90, 180, 270) and random flip(horizontal, vertical, none)
    input: Tensor, List[Tensor] or Tuple[Tensor
    note that the list of Tensor will be transformed the same way
    Tensor shape: (n, c, h, w)
    """
    x = torch.stack(x, dim=0) if isinstance(x, list) or isinstance(x, tuple) else x.unsqueeze(0)
    if not isinstance(x, Tensor):
        raise TypeError('input must be Tensor or Iterable[Tensor]')
    flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    x = torch.rot90(flip_transform(x), torch.randint(0, 4, (1,)).item(), [-2, -1])
    return x.unbind(dim=0) if x.shape[0] != 1 else x[0]


def random_crop(x: Union[Tensor, List[Tensor], Tuple[Tensor]], size: int):
    """
    random crop Tensor to size*size
    img: Tensor, List[Tensor] or Tuple[Tensor]
    size: int
    note that the list of Tensor will be cropped the same way
    Tensor shape: (n, c, h, w)
    """
    x = torch.stack(x, dim=0) if isinstance(x, list) or isinstance(x, tuple) else x.unsqueeze(0)
    crop = transforms.RandomCrop(size)
    x = crop(x)
    return x.unbind(dim=0) if x.shape[0] != 1 else x[0]
