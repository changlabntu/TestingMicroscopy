import os, glob
import tifffile as tiff
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def read_2d_tif_to_3d(xlist):
    """
    Read a list of 2D tif file and convert it to a 3D numpy array
    """
    #xlist = sorted(glob.glob(source + '/*'))
    x = [tiff.imread(x) for x in xlist]
    x = np.stack(x, 0)
    return x


def tif_to_patches(npys, **kwargs):
    """
    Convert a tif file to a folder of patches
    """
    (dz, dx, dy) = kwargs['dh']  # (64, 256, 256)
    (sz, sx, sy) = kwargs['step']

    for i in range(len(npys)):
        os.makedirs(root + kwargs['destination'][i], exist_ok=True)

    for i in range(len(npys)):
        if kwargs['trd'][i] is not None:
            npys[i][npys[i] < kwargs['trd'][i][0]] = kwargs['trd'][i][0]
            npys[i][npys[i] > kwargs['trd'][i][1]] = kwargs['trd'][i][1]

        if kwargs['norm'][i] is not None:
            if kwargs['norm'][i] == 'zrescale':
                npys[i] = z_rescale(npys[i], trd=kwargs['zrescale'])
            elif kwargs['norm'][i] == '01':
                npys[i] = (npys[i] - npys[i].min()) / (npys[i].max() - npys[i].min())
            elif kwargs['norm'][i] == '11':
                npys[i] = (npys[i] - npys[i].min()) / (npys[i].max() - npys[i].min())
                npys[i] = (npys[i] - 0.5) * 2


    for z in range(npys[0].shape[0] // sz):
        for x in range(npys[0].shape[1] // sx):
            for y in range(npys[0].shape[2] // sy):
                    volumes = []
                    for i in range(0, len(npys)):
                        volumes.append(npys[i][z * dz : (z+1) * dz, x * dx : (x+1) * dx, y * dy : (y+1) * dy])

                    if volumes[0].shape == (dz, dx, dy):
                        print(volumes[0].mean())
                        if volumes[0].mean() > kwargs['ftr']:
                            for i in range(0, len(npys)):
                                if kwargs['permute'] is not None:
                                    volumes[i] = np.transpose(volumes[i], kwargs['permute'])
                                volume = volumes[i].astype(np.float32)
                                print(volume.shape)
                                tiff.imwrite(
                                    root + kwargs['destination'][i] + kwargs['prefix'] + str(x).zfill(3) + str(
                                        y).zfill(3) + str(z).zfill(3) + '.tif', volume)



def resampling(source, destination, scale=None, size=None):
    """
    Resampling a tif file
    """
    x = tiff.imread(source)

    if scale is not None:
        up = nn.Upsample(scale_factor=scale, mode='trilinear')
    if size is not None:
        for i in range(len(size)):
            if size[i] == -1:
                size[i] = x.shape[i]
        up = nn.Upsample(size=size, mode='trilinear')

    dtype = x.dtype
    x = x.astype(np.float32)
    out = up(torch.from_numpy(x).unsqueeze(0).unsqueeze(0))
    tiff.imwrite(destination, out[0, 0, :, :, :].numpy().astype(dtype))


def z_rescale(xx, trd=6):
    xx=np.log10(xx+1);xx=np.divide((xx-xx.mean()), xx.std());
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd;
    return xx


if __name__ == "__main__":
    root = '/media/ghc/Ghc_data3/Chu_full_brain/VMAT_DPM/Naive_VMAT_1/'
    suffix = ''

    npy0 = tiff.imread(root + 'mask_cropped.tif')
    npy1 = tiff.imread(root + 'raw_cropped.tif')
    tif_to_patches([npy0, npy1],
                   destination=['maskpatch/', 'oripatch/'],
                   dh=(48, 384, 384), step=(48, 384, 384), permute=None,
                   trd=((0, 255), (90, 360)), norm=('11', '11'),
                   prefix='naivevmat1', ftr=-0.95, zrescale=None)
