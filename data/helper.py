import os
import torch
import pydicom
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor


class Helper(ABC):
    @abstractmethod
    def __len__(self,) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Tuple[Tensor]:
        """
        return input, target pair
        """
        pass


class Mayo(Helper):
    def __init__(self, root: str):
        info = []
        for root, _, files in os.walk(root):
            for file in files:
                if file.endswith('.IMA'):
                    patient, dose, _, _ = file.split('_')
                    index = file.split('.')[3]
                    info.append((patient, dose, index, os.path.join(root, file)))
        fd_dict = {i[3]: i[:3] for i in info if i[1] == 'FD'}
        qd_vdict = {i[:3]: i[3] for i in info if i[1] == 'QD'}
        self.path_pairs = [(qd_vdict[(v[0], 'QD', v[2])], k) for k, v in fd_dict.items()]

    def __getitem__(self, index: int):
        path_pair = self.path_pairs[index]
        qd = pydicom.dcmread(path_pair[0]).pixel_array.astype(np.float32)
        fd = pydicom.dcmread(path_pair[1]).pixel_array.astype(np.float32)
        qd = torch.from_numpy(qd)
        fd = torch.from_numpy(fd)
        return qd, fd

    def __len__(self):
        return len(self.path_pairs)
