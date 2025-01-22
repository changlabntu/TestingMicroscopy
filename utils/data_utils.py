import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x

# def to_16bit(x, lower_bound=0, upper_bound=550):
#     scaled = (img_array - scale_min) / (scale_max - scale_min)


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    x = x - x.min()
    x = Image.fromarray(to_8bit(x))

    if show:
        io.imshow(np.array(x))
        plt.show()
    if save:
        x.save(save)


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x

def purge_logs():
    import glob, os
    list_version = sorted(glob.glob('logs/default/*/'))
    list_checkpoint = sorted(glob.glob('logs/default/*/checkpoints/*'))

    checkpoint_epochs = [0] * len(list_version)
    for c in list_checkpoint:
        checkpoint_epochs[list_version.index(c.split('checkpoints')[0])] = int(c.split('epoch=')[-1].split('.')[0])

    for i in range(len(list_version)):
        if checkpoint_epochs[i] < 60:
            os.system('rm -rf ' + list_version[i])


class DataNormalization:
    def __init__(self, backward_type="float32"):
        self.backward_type = backward_type
        assert self.backward_type in ["float32", "uint16", "uint8"]

    def forward_normalization(self, x0, norm_method, exp_trd, exp_ftr, trd):
        if norm_method == 'exp':
            x0[x0 <= exp_trd[0]] = exp_trd[0]
            x0[x0 >= exp_trd[1]] = exp_trd[1]
            x0 = np.log10(x0 + 1)
            x0 = np.divide((x0 - x0.mean()), x0.std())
            x0[x0 <= -exp_ftr] = -exp_ftr
            x0[x0 >= exp_ftr] = exp_ftr
            x0 = x0 / exp_ftr
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '11':
            x0[x0 <= trd[0]] = trd[0]
            x0[x0 >= trd[1]] = trd[1]
            # x0 = x0 / x0.max()
            x0 = (x0 - x0.min()) / (x0.max() - x0.min())
            x0 = (x0 - 0.5) * 2
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '00':
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '01':
            print(trd[1])
            x0[x0 >= trd[1]] = trd[1]
            # x0 = x0 / x0.max()
            x0 = (x0 - x0.min()) / (x0.max() - x0.min())
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        return x0

    def backward_normalization(self, x0, norm_method, exp_trd, trd):
        x0 = self._reverse_normalization(x0, norm_method)
        if self.backward_type == "float32":
            return x0
        elif self.backward_type == "uint16":
            if norm_method == 'exp':
                lower_bound, upper_bound = exp_trd[0], exp_trd[1]
            elif norm_method in ['11', '00', '01']:
                lower_bound, upper_bound = trd[0], trd[1]
            else:
                lower_bound, upper_bound = 0, 550
            return self.to_16bit(x0, lower_bound, upper_bound)
        elif self.backward_type == "uint8":
            return self.to_8bit(x0)

    def _reverse_normalization(self, x0, norm_method):
        if norm_method == 'exp':
            return np.power(10, x0)
        elif norm_method == '11':
            x0[x0 <= -1] = -1
            x0[x0 >= 1] = 1
            x0 = (x0 + 1) / 2
            # x0 = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-7)
            return x0
        elif norm_method == '00':
            return x0
        elif norm_method == '01':
            x0[x0 <= 0] = 0
            x0[x0 >= 1] = 1
            return x0

    def to_8bit(self, x0):
        if type(x0) == torch.Tensor:
            x0 = (x0 / x0.max() * 255).numpy().astype(np.uint8)
        else:
            x0 = (x0 / x0.max() * 255).astype(np.uint8)
        return x0

    def to_16bit(self, x0, lower_bound=0, upper_bound=550):
        if type(x0) == torch.Tensor:
            x0 = x0.numpy()
        x0 = x0 * (upper_bound - lower_bound) + lower_bound
        return x0.astype(np.uint16)
