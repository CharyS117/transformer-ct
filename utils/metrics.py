import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, tensor


def psnr(x1: Tensor, x2: Tensor, bits=8):
    if not isinstance(x1, Tensor) or not isinstance(x2, Tensor):
        raise TypeError('x1 and x2 must be torch.Tensor')
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same shape')
    i_max = tensor((2.,), device=x1.device) ** bits - 1
    mse = F.mse_loss(x1, x2)
    return 10 * torch.log10(i_max ** 2 / (mse + 1e-5))


def ssim(x1: Tensor, x2: Tensor, window_size: int = 11, size_average: bool = True):
    if not isinstance(x1, Tensor) or not isinstance(x2, Tensor):
        raise TypeError('x1 and x2 must be torch.Tensor')
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same shape')
    channel = x1.shape[1]
    win = init_window(window_size, channel)
    win = win.to(x1.get_device())
    win = win.type_as(x1)
    return _ssim(x1, x2, win, window_size, channel, size_average)


def init_window(window_size, channel):
    # gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    win_1d = gauss.unsqueeze(1)
    win_2d = win_1d.mm(win_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = win_2d.expand(channel, 1, window_size, window_size)
    return window


def _ssim(x1, x2, window, window_size, channel, size_average=True):
    mean1 = F.conv2d(x1, window, padding=window_size // 2, groups=channel)
    mean2 = F.conv2d(x2, window, padding=window_size // 2, groups=channel)

    mean1_2 = mean1.pow(2)
    mean2_2 = mean2.pow(2)
    mean1mean2 = mean1 * mean2

    var1 = F.conv2d(x1 * x1, window, padding=window_size // 2, groups=channel) - mean1_2
    var2 = F.conv2d(x2 * x2, window, padding=window_size // 2, groups=channel) - mean2_2
    cov = F.conv2d(x1 * x2, window, padding=window_size // 2, groups=channel) - mean1mean2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mean1mean2 + c1) * (2 * cov + c2)) / ((mean1_2 + mean2_2 + c1) * (var1 + var2 + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
