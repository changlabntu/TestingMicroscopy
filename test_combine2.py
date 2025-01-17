import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import time
import shutil
from tqdm import tqdm
import json
import glob
import tifffile as tiff
import yaml
from tqdm import tqdm
from utils.model_utils import read_json_to_args, import_model, load_pth
import argparse
import ipdb


def get_gan_out(x0, model):
    #x0 = [x.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2) for x in x0]
    #x0 = torch.cat(x0, dim=1).cuda()  # concatenate all the input channels
    # (Z, C, X, Y)
    x0 = x0.permute(1, 2, 3, 0).unsqueeze(0)
    if gpu:
        x0 = x0.cuda()

    if 0:
        z = model(x0, method='encode')[-1].detach().cpu()
        if gpu:
            z = z.cuda()
        XupX = model(z, method='decode')['out0'].detach().cpu()
        XupX = XupX[0, 0, :, :, :]  # (X, Y, Z)

    XupX = model(x0)['out0'].detach().cpu()[0, :, :, :, :]   # out0 for original, out1 for psuedo-mask

    Xup = x0.detach().cpu().squeeze(0)#.numpy()

    XupX = XupX.permute(3, 0, 1, 2)#torch.transpose(XupX, (2, 0, 1))  # (Z, X, Y)
    Xup = Xup.permute(3, 0, 1, 2)#torch.transpose(Xup, (2, 0, 1))

    return XupX, Xup

import time

def get_ae_out(x0, model):
    hbranchz = args.hbranchz
    if gpu:
        x0 = x0.cuda()
    hb_all = []
    for z in range(0, 32, 4):
        reconstructions, posterior, hbranch = model.forward(x0[z:z+4, :, :, :], sample_posterior=False)
        if hbranchz:
            hb_all.append(posterior.sample())
        else:
            hb_all.append(hbranch)
    hbranch = torch.cat(hb_all, dim=0).detach().cpu()
    del hb_all

    if hbranchz:
        hbranch = model.decoder.conv_in(hbranch.cuda(0))

    # (Z, C, X, Y)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
    if gpu:
        hbranch = hbranch.cuda()

    XupX_all = []
    for aug in [None, 'transpose', 'flip2', 'flip3'][:]:
        hbranchP = test_time_augementation(hbranch, method=aug)

        XupX = model.net_g(hbranchP, method='decode')['out0']  # (1, C, X, Y, Z)

        XupX = test_time_augementation(XupX, method=aug)
        XupX_all.append(XupX)
    XupX = torch.mean(torch.stack(XupX_all, 0), 0)

    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
    Xup = Xup[0, :, ::].permute(3, 0, 1, 2).detach().cpu()#.numpy()  # (Z, C, X, Y))

    XupX = XupX[0, :, ::].permute(3, 0, 1, 2).detach().cpu()#.numpy()

    return XupX, Xup


def test_model(x0, model, input_augmentation=None, **kwargs):
    out_all = []
    for m in range(mc):
        d0 = kwargs['patch_range']['d0']
        dx = kwargs['patch_range']['dx']
        patch = [x[:, :, d0[0]:d0[0]+dx[0], d0[1]:d0[1]+dx[1], d0[2]:d0[2]+dx[2]] for x in x0]

        patch = torch.cat([upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

        #out_aug = []
        for i, aug in enumerate(input_augmentation):  # (Z, C, X, Y)
            # reshape to 3d for augmentation
            input = 1 * patch
            input = input.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
            # augmentation
            input = test_time_augementation(input, method=aug)
            # reshape back to 2d for input
            input = input.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

            # here is the forward
            if model_type == 'GAN':
                out, Xup = get_gan_out(input, model)
            elif model_type == 'AE':
                out, Xup = get_ae_out(input, model)   # (Z, X, Y)

            # reshape to 3d for augmentation
            # (Z, C, X, Y)

            out = out.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
            Xup = Xup.permute(1, 2, 3, 0).unsqueeze(0)
            # augmentation back
            out = test_time_augementation(out, method=aug)
            Xup = test_time_augementation(Xup, method=aug)
            # reshape back to 2d for input
            out = out.squeeze(0).permute(0, 3, 1, 2)
            Xup = Xup.squeeze(0).permute(0, 3, 1, 2)

            out_all.append(out.numpy())

            #out_aug.append(out)

        #out_aug = torch.stack(out_aug, 0)
        #out = torch.mean(out_aug, 0)

        out_all.append(out.numpy())

    out_all = np.stack(out_all, axis=-1)

    return out_all, Xup.numpy()


def reverse_log(x):
    return np.power(10, x)


def assemble_microscopy_volumne(kwargs, zrange, xrange, yrange, source):
    C0, C1, C2 = kwargs['assemble_params']['C']
    S0, S1, S2 = kwargs['assemble_params']['S']
    # C = kwargs['assemble_params']['C']
    # S = kwargs['assemble_params']['S']

    one_stack = []
    for nx in range(len(xrange)):
        one_column = []
        for nz in range(len(zrange)):
            one_row = []
            for ny in range(len(yrange)):
                # get weight
                if nx == len(xrange) - 1:
                    nx = -1
                if ny == len(yrange) - 1:
                    ny = -1
                if nz == len(zrange) - 1:
                    nz = -1

                iz = zrange[nz]
                ix = xrange[nx]
                iy = yrange[ny]

                w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=kwargs['assemble_params']['weight_shape'], edge_size=64)

                # load and crop
                x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')[0, ::]

                if C0 > 0:
                    cropped = x[C0:-C0, :, :]
                if C1 > 0:
                    cropped = cropped[:, C1:-C1, :]
                if C2 > 0:
                    cropped = cropped[:, :, C2:-C2]
                # ipdb.set_trace()
                cropped = np.multiply(cropped, w)
                if len(one_row) > 0:
                    one_row[-1][:, :, -S2:] = one_row[-1][:, :, -S2:] + cropped[:, :, :S2]
                    one_row.append(cropped[:, :, S2:])
                else:
                    one_row.append(cropped)

            one_row = np.concatenate(one_row, axis=2)  # (Z, X, Y)
            one_row = np.transpose(one_row, (1, 0, 2))  # (X, Z, Y)

            if len(one_column) > 0:
                one_column[-1][:, -S0:, :] = one_column[-1][:, -S0:, :] + one_row[:, :S0, :]
                one_column.append(one_row[:, S0:, :])
            else:
                one_column.append(one_row)

        one_column = np.concatenate(one_column, axis=1).astype(np.float32)

        # ipdb.set_trace()
        tiff.imwrite('o.tif', one_column)

        if len(one_stack) > 0:
            one_stack[-1][-S1:, :, :] = one_stack[-1][-S1:, :, :] + one_column[:S1, :, :]
            one_stack.append(one_column[S1:, :, :])
        else:
            one_stack.append(one_column)

    one_stack = np.concatenate(one_stack, axis=0).astype(np.float32)  # (X, Z, Y)

    one_stack = np.transpose(one_stack, (2, 1, 0))

    print(source)
    tiff.imwrite(source[:-1] + '.tif', one_stack)


def test_over_volumne(kwargs, dx, dy, dz, zrange, xrange, yrange, destination, input_augmentation):
    for ix in xrange:
        for iz in tqdm(zrange):
            for iy in yrange:
                kwargs['patch_range']['d0'] = [iz, ix, iy]
                kwargs['patch_range']['dx'] = [dz, dx, dy]

                out, patch = test_model(x0, model, input_augmentation=input_augmentation, **kwargs)

                #if mc > 1:
                var = ((out > -0.16) / 1).std(axis=-1)
                out = out.mean(axis=-1).astype(np.float32)

                if args.reverselog:
                    out = reverse_log(out)
                    patch = reverse_log(patch)

                tiff.imwrite(destination + 'xy/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out)
                tiff.imwrite(destination + 'ori/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', patch)

                #if mc > 1:
                tiff.imwrite(destination + 'xyvar/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', var)

                #if mc > 1:
                #    tiff.imwrite(destination + 'xyvar/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.std(axis=3).astype(np.float32))


def get_model(dataset, prj, epoch, model_type, gpu):
    #dataset = kwargs['dataset']
    #prj = kwargs['prj']
    #epoch = kwargs['epoch']

    # get GAN model
    if model_type == 'GAN':
        model_name = path_source + '/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth'
        print(model_name)
        model = torch.load(model_name, map_location=torch.device('cpu'))

    # get AE model
    if model_type == 'AE':
        component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
        root = path_source + '/logs/' + dataset + prj
        args = read_json_to_args(root + '0.json') # load config json file

        # dynamically load module
        model_module = import_model(root, model_name=args.models)
        model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
        model = load_pth(model, root=root, epoch=epoch, model_names=component_names)

    #if model_type == 'GAN':
    upsample = torch.nn.Upsample(size=kwargs['upsample_params']['size'], mode='trilinear')
    #else:
    #    upsample = torch.nn.Upsample((32, 256, 256), mode='trilinear')
    if gpu:
        model = model.cuda()
        upsample = upsample.cuda()

    for param in model.parameters():
        param.requires_grad = False

    return model, upsample


def recreate_volume_folder(destination):
    # remove and recreate the folder
    if os.path.exists(destination + 'xy/'):
        shutil.rmtree(destination + 'xy/')
    if os.path.exists(destination + 'ori/'):
        shutil.rmtree(destination + 'ori/')
    os.makedirs(destination + 'xy/', exist_ok=True)
    os.makedirs(destination + 'ori/', exist_ok=True)
    #if mc > 1:
    os.makedirs(destination + 'xyvar/', exist_ok=True)


def view_two_other_direction(x):
    return np.concatenate([np.transpose(x, (2, 1, 0)), np.transpose(x, (1, 2, 0))], 2)


def slice_for_ganout():
    rois = sorted(glob.glob(path_source + '/Dataset/paired_images/' + kwargs["dataset"] + '/cycout/xy/*.tif'))

    for roi in tqdm(rois[:]):
        xy = tiff.imread(roi)
        ori = tiff.imread(roi.replace('/xy/', '/ori/'))

        xy = xy[64:-64, 32:-32, 64:-64]
        ori = ori[64:-64, 32:-32, 64:-64]

        if xy.mean() >= -0.5:
            for ix in range(xy.shape[1]):
                tiff.imwrite(roi.replace('/xy/', '/ganxy/')[:-4] + '_' + str(ix).zfill(3) + '.tif', xy[:, ix, :])
                tiff.imwrite(roi.replace('/xy/', '/ganori/')[:-4] + '_' + str(ix).zfill(3) + '.tif', ori[:, ix, :])


def create_tapered_weight(S0, S1, S2, nz, nx, ny, size, edge_size: int = 64) -> np.ndarray:
    """
    Create a 3D cube with linearly tapered edges in all directions.

    Args:
        size (int): Size of the cube (size x size x size)
        edge_size (int): Size of the tapered edge section

    Returns:
        np.ndarray: 3D array with tapered weights
    """
    # Create base cube filled with ones
    weight = np.ones(size)

    # Create linear taper from 0 to 1
    # taper = np.linspace(0, 1, edge_size)
    taper_S0 = np.linspace(0, 1, S0)
    taper_S1 = np.linspace(0, 1, S1)
    taper_S2 = np.linspace(0, 1, S2)

    # Z
    if nz != 0:
        weight[:S0, :, :] *= taper_S0.reshape(-1, 1, 1)
    if nz != -1:
        weight[-S0:, :, :] *= taper_S0[::-1].reshape(-1, 1, 1)

    # X
    if nx != 0:
        weight[:, :S1, :] *= taper_S1.reshape(1, -1, 1)
    if nx != -1:
        weight[:, -S1:, :] *= taper_S1[::-1].reshape(1, -1, 1)

    # Y
    if ny != 0:
        weight[:, :, :S2] *= taper_S2
    if ny != -1:
        weight[:, :, -S2:] *= taper_S2[::-1]

    return weight


def get_args(option, config_name):
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)

    kwargs = config.get(option, {})
    if not kwargs:
        raise ValueError(f"Option {option} not found in the configuration.")
    return config, kwargs


def get_data(kwargs):
    image_path = [kwargs.get("root_path") + x for x in kwargs.get("image_path")]  # if image path is a file
    image_list_path = kwargs.get("image_list_path")  # if image path is a directory

    x0 = []
    if image_path:
        for i in range(len(image_path)):
            x0.append(tiff.imread(image_path[i]))

    elif image_list_path:
        for i in range(len(image_list_path)):
            x_list = sorted(glob.glob(image_list_path[i]))
            if not x_list:
                raise ValueError(f"No images found at {image_list_path[i]}")
            x0.append(tiff.imread(x_list[kwargs.get("image_list_index")]))
    else:
        raise ValueError("No valid image path provided.")
    return x0


def norm_x0(x0, norm_method, exp_trd, exp_ftr, trd):
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
        if trd[0] is None:
            trd[0] = x0.min()
        x0[x0 <= trd[0]] = trd[0]
        x0[x0 >= trd[1]] = trd[1]
        #x0 = x0 / x0.max()
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x0 = (x0 - 0.5) * 2
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    elif norm_method == '00':
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    elif norm_method == '01':
        print(trd[1])
        x0[x0 >= trd[1]] = trd[1]
        #x0 = x0 / x0.max()
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    return x0


def test_time_augementation(x, method):
    # x shape: (1, C, X, Y, Z)
    if method == None:
        return x
    elif method.startswith('flip'):
        x = torch.flip(x, dims=[int(method[-1])])
        return x
    elif method == 'transpose':
        x = x.permute(0, 1, 3, 2, 4)
        return x


def test_args():
    parser = argparse.ArgumentParser()
    # projects
    parser.add_argument('--config', type=str, default="config", help='which config file')
    parser.add_argument('--option', type=str, default="Default", help='which dataset to use')
    parser.add_argument('--prj', type=str, default="/ae/cut/1/", help='name of the project')
    parser.add_argument('--epoch', type=str, default='3000', help='epoch #')
    parser.add_argument('--model_type', type=str, default='AE', help='GAN or AE')
    parser.add_argument('--testvolume', action='store_true', default=False)
    parser.add_argument('--assemble', action='store_true', default=False)
    parser.add_argument('--hbranchz', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--reverselog', action='store_true', default=False)
    return parser


if __name__ == '__main__':
    # Model parameters
    parser = test_args()
    args = parser.parse_args()
    # Data parameters
    config, kwargs = get_args(option=args.option, config_name='test/' + args.config + '.yaml')
    print(kwargs)

    model_type = args.model_type
    path_source = config['SOURCE']
    destination = kwargs['root_path']

    gpu = args.gpu
    mc = kwargs['mc']  # monte carlo inference, mean over N times

    # model
    model, upsample = get_model(kwargs['dataset'], args.prj, args.epoch, args.model_type, gpu)

    # Data
    x0 = get_data(kwargs)
    for i in range(len(x0)):
        x0[i] = norm_x0(x0[i], kwargs['norm_method'][i],
                        kwargs['exp_trd'][i], kwargs['exp_ftr'][i], kwargs['trd'][i])

    # single test
    out, patch = test_model(x0, model, input_augmentation=[None, 'transpose', 'flip2', 'flip3'][:1], **kwargs)
    var = ((out > -0.16) / 1).std(axis=-1)
    #var = out.std(axis=-1)
    out = out.mean(axis=-1)

    # save single output
    if args.reverselog:
        out = reverse_log(out)
        patch = reverse_log(patch)

    print((out.shape, patch.shape))

    tiff.imwrite(destination + '/xy.tif', np.transpose(out, (0, 2, 1, 3)))
    tiff.imwrite(destination + '/var.tif', np.transpose(var, (0, 2, 1, 3)))
    tiff.imwrite(destination + '/patch.tif', np.transpose(patch, (0, 2, 1, 3)))

    # assembly test

    dz, dx, dy = kwargs['assemble_params']['dx_shape']
    zrange = range(*kwargs['assemble_params']['zrange'])
    xrange = range(*kwargs['assemble_params']['xrange'])
    yrange = range(*kwargs['assemble_params']['yrange'])

    if args.testvolume:
        recreate_volume_folder(destination + '/cycout/')  # DELETE and recreate the folder
        test_over_volumne(kwargs, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                          destination=destination + '/cycout/', input_augmentation=[None, 'transpose', 'flip2', 'flip3'][:1])

    if args.assemble:
        assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/xy/')

        assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/ori/')

        assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/xyvar/')



    # USAGE
    # Fly0B (This is the "10X" fly data):
    #   GAN:
    #   python test_combine.py  --prj /IsoScopeXXcut/ngf32lb10/ --epoch 5000 --model_type GAN --gpu
    #   AE: (This on has "--hbranchz")
    #   python test_combine.py  --prj /ae/cut/1/ --epoch 3000 --model_type AE --gpu --hbranchz

    # DPM4X: (This is the main data)
    #   python test_combine.py  --prj /ae/iso0_ldmaex2_lb10_tc/ --epoch 2300 --model_type AE --gpu --reverselog --assemble
    #   python test_combine.py  --prj /ae/cut/1/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --assemble

    #   python test_combine.py  --prj /IsoScopeXY16X/ngf32ndf32lb10skip2nocyc --epoch 3000 --model_type GAN --gpu --config weikun060524

