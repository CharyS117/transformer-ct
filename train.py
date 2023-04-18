import yaml
import torch
import os
import logging
import argparse
from tqdm import tqdm
import importlib
from torch import tensor
from data.dataset import PairedDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.metrics import psnr
from utils.checkpoint import load_state, save_checkpoint, load_progress
from data.processor import random_crop, random_augmentation
from utils.logger import init_logger

# cuda settings
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = torch.device('cpu')

# parse arguments
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('config_path', help='config file path')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')
args = parser.parse_args()

# load config
with open(args.config_path) as f:
    config = yaml.safe_load(f)
    train_name = os.path.splitext(os.path.basename(args.config_path))[0]

# set up logger
init_logger(f'./log/{train_name}.log')
logger = logging.getLogger('logger')
if args.verbose:
    logger.setLevel('DEBUG')
else:
    logger.setLevel('INFO')
logger.info(f'Start training: {train_name}')
logger.debug(f'Config path: {args.config_path}')
logger.debug(f'Log path: ./log/{train_name}.log')

# set up random seed
if seed := config['train']['seed']:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.debug(f'Set random seed to {seed}')
else:
    logger.debug('No random seed set')

# set up dataloader
train_dataset = PairedDataset(name='train', **config['data']['train'])
val_dataset = PairedDataset(name='validation', **config['data']['val'])
val_dataloader = DataLoader(val_dataset)
total_iters = [(len(train_dataset)//b+1)*e for e, b in zip(config['train']['epoch'], config['train']['batch'])]
logger.debug(f'Total iters: {total_iters}')

# set up network
logger.info(f'Network: {config["net"]["module"]}')
module = importlib.import_module(f'arch.{config["net"]["module"]}')
model = module.Net(**config['net']['params']).to(device)
if config['train']['compile']:
    logger.info(f'Net compile enbled')
    model = torch.compile(model)

# set up loss
logger.info(f'Loss_fn: {config["loss_fn"]["class"]}')
module = importlib.import_module('.'.join(config['loss_fn']['class'].split('.')[:-1]))
loss_fn_class = getattr(module, config['loss_fn']['class'].split('.')[-1])
loss_fn = loss_fn_class(**config['loss_fn']['params']).to(device)

# set up optimizer
logger.info(f'Optimizer: {config["optimizer"]["class"]}')
module = importlib.import_module('.'.join(config['optimizer']['class'].split('.')[:-1]))
optimizer_class = getattr(module, config['optimizer']['class'].split('.')[-1])
optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])

# load checkpoint
checkpoint_dir = os.path.join(config['train']['checkpoint_root'], train_name)
checkpoint_name = config['train']['checkpoint_name']
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name) if checkpoint_name else None
checkpoint_interval = config['train']['checkpoint_interval']
os.makedirs(checkpoint_dir, exist_ok=True)
start_stage, start_epoch, best_val, loss_history = load_progress(checkpoint_path)
scheduler = CosineAnnealingLR(optimizer, T_max=total_iters[start_stage], eta_min=config['train']['scheduler_eta_min'], verbose=args.verbose)
load_state(model, optimizer, scheduler, checkpoint_path)

# train
total_stage = len(config['train']['epoch'])
bits_rate = 2**(config['data']['image']['to_bits'] - config['data']['image']['from_bits'])
enable_amp = config['train']['enable_amp']
if enable_amp:
    logger.info('AMP enabled')
for stage, epoch_num in enumerate(config['train']['epoch']):
    # iterate to loaded stage
    if stage < start_stage:
        continue
    # set up dataloader
    batch_size = config['train']['batch'][stage]
    img_size = config['train']['size'][stage]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, **config['dataloader'])
    logger.info(f'Start stage {stage+1}, epoch_num {epoch_num}, img_size {img_size}, batch_size {batch_size}')
    for epoch in range(epoch_num):
        # iterate to loaded epoch
        if epoch <= start_epoch:
            continue
        start_epoch = -1
        model.train()
        loss_record = tensor([0.], device=device)
        epoch_loss = tensor([0.], device=device)
        train_loop = tqdm(total=len(train_dataloader), desc=f'Stage: {stage+1}/{total_stage} Epoch: {epoch+1}/{epoch_num}', ncols=150, leave=False)
        for iters, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [i.to(device)*bits_rate for i in batch]
            batch = random_augmentation(batch)
            low, high = random_crop(batch, img_size)
            if enable_amp:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    output = model(low)
                    loss = loss_fn(output, high)
            else:
                output = model(low)
                loss = loss_fn(output, high)
            loss.backward()
            if clip_norm := config['train']['grad_clip_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            loss_record += loss
            epoch_loss += loss
            loss_record_interval = config['train']['loss_record_interval']
            if iters % loss_record_interval == loss_record_interval - 1:
                avg_record_loss = (loss_record/loss_record_interval).item()
                loss_history.append(avg_record_loss)
                train_loop.set_postfix(loss=avg_record_loss)
                loss_record = tensor((0.,), device=device)
            scheduler.step()
            train_loop.update()
        train_loop.close()
        logger.debug(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
        # validation
        model.eval()
        avg_psnr = tensor((0.,), device=device)
        avg_epoch_loss = (epoch_loss / (iters + 1)).item()
        val_loop = tqdm(total=len(val_dataloader), desc=f'Stage: {stage+1}/{total_stage} Epoch: {epoch+1}/{epoch_num} Val:', ncols=150, leave=False)
        with torch.no_grad():
            for iters, batch in enumerate(val_dataloader):
                low, high = [i.to(device) * bits_rate for i in batch]
                output = model(low)
                avg_psnr += psnr(output, high, config['data']['image']['to_bits'])
                val_loop.update()
            val_loop.close()
            avg_psnr = (avg_psnr / (iters + 1)).item()
            train_time = int(train_loop.format_dict['elapsed'])
            logger.info(f'Stage {stage+1}, Epoch {epoch+1}, Val_PSNR {avg_psnr:.4f}, Avg_Loss {avg_epoch_loss:.4f}, Duration {"{:02d}:{:02d}".format(*divmod(train_time, 60))}')
            if avg_psnr > best_val:
                logger.info(f'New best')
                best_val = avg_psnr
                save_path = os.path.join(checkpoint_dir, f'{train_name}_best.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, stage, avg_psnr, loss_history, save_path)       
        # checkpoint
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_path = os.path.join(checkpoint_dir, f'{train_name}_{stage+1}_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, stage, avg_psnr, loss_history, save_path)
    # reset optimizer and scheduler
    if stage + 1 != len(config['train']['epoch']):
        optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])
        scheduler = CosineAnnealingLR(optimizer, T_max=total_iters[stage+1], eta_min=config['train']['scheduler_eta_min'], verbose=args.verbose)

# save final checkpoint
save_path = os.path.join(checkpoint_dir, f'{train_name}_final.pth')
save_checkpoint(model, optimizer, scheduler, epoch, stage, avg_psnr, loss_history, save_path)
