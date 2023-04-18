import os
import torch
import logging
from typing import Union

logger = logging.getLogger('logger')


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
        epoch: int,
        stage: int,
        val: float,
        loss_history: list,
        path: str
):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'stage': stage,
        'val': val,
        'loss_history': loss_history
    }
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(state, path)
    logger.info(f'Saved to {path}')


def load_progress(path: str):
    """
    return stage, epoch, val, loss_history
    """
    if path:
        state = torch.load(path)
        stage = state['stage']
        epoch = state['epoch']
        val = state['val'] if 'val' in state else 0
        loss_history = state['loss_history']
        logger.info(f'Load {path}: stage {stage}, epoch {epoch}, val {val}')
        return stage, epoch, val, loss_history
    else:
        logger.info('No checkpoint set, start from scratch')
        return 0, -1, 0, []


def load_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None], path: str):
    """
    load model, optimizer, scheduler state from checkpoint
    """
    if path:
        state = torch.load(path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']) if scheduler else None
