import logging
from torch.utils.data import Dataset
from data import helper as helpers
from tqdm import tqdm

logger = logging.getLogger('logger')


class PairedDataset(Dataset):
    def __init__(self, name: str, path: str, io: str, helper_name: str, load_num: int):
        helper = getattr(helpers, helper_name)(path)
        load_num = load_num if load_num else len(helper)

        if load_num > len(helper):
            logger.warning(f'Set load_num {load_num} > size {len(helper)} in {name}')
            load_num = len(helper)

        self.io = io
        if io == 'ram':
            loop = tqdm(range(load_num), desc=f'Loading {name} to RAM', leave=False)
            self.data = [helper[i] for i in loop]
            logger.info(f'Dataset {name} loaded to RAM, duration {"{:02d}:{:02d}".format(*divmod(int(loop.format_dict["elapsed"]), 60))}')
        else:
            raise NotImplementedError(f'io {io} not implemented')

    def __getitem__(self, index: int):
        return [i.unsqueeze(0) for i in self.data[index] if len(i.shape) == 2]
    
    def __len__(self):
        if self.io == 'ram':
            return len(self.data)
        else:
            # default io is ram
            return len(self.data)
        
    def __del__(self):
        if self.io == 'ram':
            del self.data
        else:
            del self.data
