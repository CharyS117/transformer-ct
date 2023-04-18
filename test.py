import torch
from arch import base
from data.dataset import PairedDataset
import matplotlib.pyplot as plt


model_80m = base.Net(48, [4,6,6,6,8])
model_20m = base.Net(48, [4,6,6,8])
model_10m = base.Net(36, [4,6,6,8])

# choose model size
model = model_80m
# set pretrained model path
pretrain_path = 'path-to-pretrained-model'
# set dataset path
dataset_path = 'path-to-dataset'


state = torch.load(pretrain_path)['model']
# remove prefix of torch.compile
compile_prefix = '_orig_mod.'
for k,v in list(state.items()):
    if k.startswith(compile_prefix):
        state[k[len(compile_prefix):]] = state.pop(k)

dataset = PairedDataset('test', dataset_path, 'ram', 'Mayo', None)
for i in len(dataset):
    qd, fd = dataset[i]
    # convert to 8 bits
    qd = qd/16
    fd = fd/16
    val_img = qd.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        model.load_state_dict(state)
        out = model(val_img)

    fd = fd[0]
    qd = qd[0]
    out = out[0][0]

    # save to png
    plt.imsave(f'output/{i+1}_qd.png', qd, cmap='gray')
    plt.imsave(f'output/{i+1}_fd.png', fd, cmap='gray')
    plt.imsave(f'output/{i+1}_out.png', out, cmap='gray')