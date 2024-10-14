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

def get_gan_out(x0, model):
    #x0 = [x.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2) for x in x0]
    #x0 = torch.cat(x0, dim=1).cuda()  # concatenate all the input channels
    # (Z, C, X, Y)
    x0 = x0.permute(1, 2, 3, 0).unsqueeze(0)
    if gpu:
        x0 = x0.cuda()

    z = model(x0, method='encode')[-1].detach().cpu()
    if gpu:
        z = z.cuda()
    XupX = model(z, method='decode')['out0'].detach().cpu()
    XupX = XupX[0, 0, :, :, :]  # (X, Y, Z)

    Xup = x0.detach().cpu().squeeze()#.numpy()

    XupX = XupX.permute(2, 0, 1)#torch.transpose(XupX, (2, 0, 1))  # (Z, X, Y)
    Xup = Xup.permute(2, 0, 1)#torch.transpose(Xup, (2, 0, 1))

    return XupX, Xup


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

    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)
    if gpu:
        hbranch = hbranch.cuda()
    XupX = model.net_g(hbranch, method='decode')['out0']  # (1, C, X, Y, Z)

    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
    Xup = Xup[0, 0, ::].permute(2, 0, 1).detach().cpu()#.numpy()  # (Z, X, Y))
    XupX = XupX[0, 0, ::].permute(2, 0, 1).detach().cpu()#.numpy()

    return XupX, Xup


def test_model(x0, model, input_augmentation=None, **kwargs):
    out_all = []
    for m in range(mc):
        patch = [x[:, :, kwargs['patch_range']['start_dim0']:kwargs['patch_range']['end_dim0'],
            kwargs['patch_range']['start_dim1']:kwargs['patch_range']['end_dim1'],
            kwargs['patch_range']['start_dim2']:kwargs['patch_range']['end_dim2']] for x in x0]

        patch = torch.cat([upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

        out_aug = []
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
            out = out.permute(1, 2, 0).unsqueeze(0).unsqueeze(1)  # (1, C, X, Y, Z)
            Xup = Xup.permute(1, 2, 0).unsqueeze(0).unsqueeze(1)
            # augmentation back
            out = test_time_augementation(out, method=aug)
            Xup = test_time_augementation(Xup, method=aug)
            # reshape back to 2d for input
            out = out.squeeze().permute(2, 0, 1)
            Xup = Xup.squeeze().permute(2, 0, 1)
            out_aug.append(out)

        out_aug = torch.stack(out_aug, 0)
        out = torch.mean(out_aug, 0)

        out_all.append(out.numpy())

    out_all = np.stack(out_all, axis=3)

    return out_all, Xup.numpy()


def reverse_log(x):
    return np.power(10, x)


def assemble_microscopy_volumne(kwargs, w, zrange, xrange, yrange, source):
    C = kwargs['assemble_params']['C']
    S = kwargs['assemble_params']['S']
    for ix in tqdm(xrange):
        one_column = []
        for iz in zrange:
            one_row = []
            for iy in yrange:
                x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                cropped = x[C:-C, :, C:-C]
                cropped = np.multiply(cropped, w)
                if len(one_row) > 0:
                    one_row[-1][:, :, -S:] = one_row[-1][:, :, -S:] + cropped[:, :, :S]
                    one_row.append(cropped[:, :, S:])
                else:
                    one_row.append(cropped)
            one_row = np.concatenate(one_row, axis=2)
            one_row = np.transpose(one_row, (1, 0, 2))

            if len(one_column) > 0:
                one_column[-1][:, -S:, :] = one_column[-1][:, -S:, :] + one_row[:, :S, :]
                one_column.append(one_row[:, S:, :])
            else:
                one_column.append(one_row)
        one_column = np.concatenate(one_column, axis=1).astype(np.float32)
    tiff.imwrite(source[:-1] + '.tif', one_column)


def test_over_volumne(kwargs, dx, dy, dz, zrange, xrange, yrange, destination, input_augmentation):
    for ix in xrange:
        for iz in tqdm(zrange):
            for iy in yrange:
                kwargs['patch_range'] = {'start_dim0': iz, 'end_dim0': iz + dz,
                                         'start_dim1': ix, 'end_dim1': ix + dx,
                                         'start_dim2': iy, 'end_dim2': iy + dy}

                out_all, patch = test_model(x0, model, input_augmentation=input_augmentation, **kwargs)

                tiff.imwrite(destination + 'xy/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.mean(axis=3).astype(np.float32))
                tiff.imwrite(destination + 'ori/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', patch)

                if mc > 1:
                    tiff.imwrite(destination + 'xyvar/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.std(axis=3).astype(np.float32))


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

    if model_type == 'GAN':
        upsample = torch.nn.Upsample(size=kwargs['upsample_params']['size'], mode='trilinear')
    else:
        upsample = torch.nn.Upsample((32, 256, 256), mode='trilinear')
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
    if mc > 1:
        os.makedirs(destination + 'xyvar/')


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


def get_weight(size, method='cross', S=32):
    # the linearly tapering weight to combine al the individual ROI
    weight = np.ones(size)
    weight[:, :, :S] = np.linspace(0, 1, S)
    weight[:, :, -S:] = np.linspace(1, 0, S)
    if method == 'row':
        return weight
    if method == 'cross':
        weight = np.multiply(np.transpose(weight, (2, 1, 0)), weight)
        return weight


def get_args(option, config_name):
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)

    kwargs = config.get(option, {})
    if not kwargs:
        raise ValueError(f"Option {option} not found in the configuration.")
    return config, kwargs


def get_data(kwargs):
    image_path = [path_source + x for x in kwargs.get("image_path")]  # if image path is a file
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
    parser.add_argument('--option', type=str, default="Fly0B", help='which dataset to use')
    parser.add_argument('--prj', type=str, default="/ae/cut/1/", help='name of the project')
    parser.add_argument('--epoch', type=str, default='3000', help='epoch #')
    parser.add_argument('--model_type', type=str, default='AE', help='GAN or AE')
    parser.add_argument('--assemble', action='store_true', default=False)
    parser.add_argument('--hbranchz', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    return parser


if __name__ == '__main__':
    # Model parameters
    parser = test_args()
    args = parser.parse_args()
    # Data parameters
    config, kwargs = get_args(option=args.option, config_name='test/config.yaml')
    print(kwargs)

    model_type = args.model_type
    path_source = config['SOURCE']
    destination = path_source + '/Dataset/paired_images/' + kwargs["dataset"]

    gpu = args.gpu
    mc = 1  # monte carlo inference, mean over N times

    # model
    model, upsample = get_model(kwargs['dataset'], args.prj, args.epoch, args.model_type, gpu)

    # Data
    x0 = get_data(kwargs)
    for i in range(len(x0)):
        x0[i] = norm_x0(x0[i], kwargs['norm_method'][i],
                        kwargs['exp_trd'][i], kwargs['exp_ftr'][i], kwargs['trd'][i])

    # single test
    out, patch = test_model(x0, model, input_augmentation=[None, 'transpose', 'flip2', 'flip3'][:], **kwargs)
    out = out.mean(axis=3)

    # save single output
    tiff.imwrite(destination + '/xy.tif', np.transpose(out, (1, 0, 2)))
    tiff.imwrite(destination + '/patch.tif', np.transpose(patch, (1, 0, 2)))

    # assembly test
    if args.assemble:
        dz, dx, dy = kwargs['assemble_params']['dx_shape']
        w = get_weight(kwargs['assemble_params']['weight_shape'], method='cross', S=kwargs['assemble_params']['S'])
        zrange = range(*kwargs['assemble_params']['zrange'])
        xrange = range(*kwargs['assemble_params']['xrange'])
        yrange = range(*kwargs['assemble_params']['yrange'])

        recreate_volume_folder(destination + '/cycout/')  # DELETE and recreate the folder
        test_over_volumne(kwargs, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                          destination=destination + '/cycout/', input_augmentation=[None, 'transpose', 'flip2', 'flip3'][:])

        assemble_microscopy_volumne(kwargs, w, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/xy/')

        assemble_microscopy_volumne(kwargs, w, zrange=zrange, xrange=xrange, yrange=yrange,
                                    source=destination + '/cycout/ori/')


    # USAGE
    # DPM4X:
    # python test_combine.py  --prj /ae/iso0_ldmaex2_lb10_tc/ --epoch 2300 --model_type AE --option DPM4X --gpu --assemble
    # Fly0B:
    #   GAN:
    #   python test_combine.py  --prj /IsoScopeXXcut/ngf32lb10/ --epoch 5000 --model_type GAN --option Fly0B --gpu
    #   AE: (This on has "--hbranchz")
    #   python test_combine.py  --prj /ae/cut/1/ --epoch 3000 --model_type AE --option Fly0B --gpu --hbrahcnz


