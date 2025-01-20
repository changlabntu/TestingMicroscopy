import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse
import os, importlib, sys

import torch.nn as nn


def load_pth(gan, root, epoch, model_names):
    for name in model_names:
        setattr(gan, name, torch.load(root + 'checkpoints/' + name + '_model_epoch_' + str(epoch) + '.pth',
                                      map_location=torch.device('cpu')))
    return gan


def import_model(root, model_name):
    model_path = os.path.join(root, f"{model_name}.py")
    module_name = f"dynamic_model_{model_name}"

    # Create the spec
    spec = importlib.util.spec_from_file_location(module_name, model_path)

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    return module


# class Sequential2DUpsampling(nn.Module):
#     """
#     Replace 3D upsampling with two sequential 2D upsampling operations.
#     Assumes input shape is (batch, channel, depth, height, width)
#     """
#
#     def __init__(self, scale_factor=(2, 2, 2)):
#         super().__init__()
#         self.scale_factor = scale_factor
#
#         # Create 2D upsampling for spatial dimensions (H, W)
#         self.spatial_upsample = nn.Upsample(
#             scale_factor=(self.scale_factor[1], self.scale_factor[2]),
#             mode='nearest'
#         )
#
#         # Create 2D upsampling for depth dimension (D, W)
#         self.depth_upsample = nn.Upsample(
#             scale_factor=(self.scale_factor[0], 1),
#             mode='nearest'
#         )
#
#     def forward(self, x):
#         # Input shape: (B, C, D, H, W)
#         B, C, D, H, W = x.shape
#
#         # Step 1: Handle spatial dimensions (H, W)
#         # Reshape to treat depth as batch dimension
#         x = x.transpose(1, 2).reshape(B * D, C, H, W)
#         print('xxx')
#         print(x.shape)
#
#
#         x = self.spatial_upsample(x)
#
#         # Get new spatial dimensions
#         _, _, H_new, W_new = x.shape
#
#         # Step 2: Handle depth dimension
#         # Reshape back to 5D, putting width dimension last
#         x = x.reshape(B, D, C, H_new, W_new).transpose(1, 3)  # (B, D, C, H_new, W_new) -> (B, D, H_new, C, W_new)
#         # Reshape to treat it as 2D upsampling problem for depth
#         x = x.reshape(B * H_new, C, D, W_new)
#
#         print('hi')
#         print(x.shape)
#
#         x = self.depth_upsample(x)
#
#         # Get new depth dimension
#         _, _, D_new, _ = x.shape
#
#         # Reshape back to original 5D format
#         x = x.reshape(B, H_new, C, D_new, W_new)
#         x = x.permute(0, 2, 3, 1, 4)  # (B, C, D_new, H_new, W_new)
#
#         return x


def read_json_to_args(json_file):
    with open(json_file, 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    return args

class ModelProcesser:
    def __init__(self, args, model, upsample_size = None):
        self.args = args
        self.model = model
        self.upsample_size = upsample_size
        self.gpu = args.gpu
        # dummy method for fix upsample bug
        # self.model.net_g.decoder[0][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))
        # self.model.net_g.decoder[2][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))
        # self.model.net_g.decoder[4][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))

    def get_model_result(self, x0):
        if self.args.model_type == 'AE':
            XupX, Xup, XupX_seg = self.get_ae_out(x0)
        elif self.args.model_type == 'GAN':
            XupX, Xup, XupX_seg = self.get_gan_out(x0)
        return XupX, Xup, XupX_seg

    def get_gan_out(self, x0):
        # x0 = [x.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2) for x in x0]
        # x0 = torch.cat(x0, dim=1).cuda()  # concatenate all the input channels
        # (Z, C, X, Y)
        x0 = x0.permute(1, 2, 3, 0).unsqueeze(0)
        if self.gpu:
            x0 = x0.cuda()

        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            out = self.model(x0)
            XupX = out['out0'].detach().cpu()[0, 0, :, :, :]  # out0 for original, out1 for psuedo-mask

        Xup = x0.detach().cpu().squeeze()  # .numpy()

        XupX = XupX.permute(2, 0, 1)  # torch.transpose(XupX, (2, 0, 1))  # (Z, X, Y)
        Xup = Xup.permute(2, 0, 1)  # torch.transpose(Xup, (2, 0, 1))

        if "seg" in self.args.save:
            XupX_seg = out['out1'].detach().cpu()
            XupX_seg = XupX_seg[0, 0, ::].permute(2, 0, 1)
            return XupX, Xup, XupX_seg
        return XupX, Xup, "_"

    def get_ae_out(self, x0):
        reconstructions, Xup, hbranch = self.get_ae_encode(x0)
        XupX, XupX_seg = self.get_ae_decode(hbranch)

        return XupX, Xup, XupX_seg

    def get_ae_encode(self, x0):
        if self.gpu:
            x0 = x0.cuda()
        hb_all = []
        reconstructions_all = []
        for z in range(0, x0.shape[0], 4):
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                reconstructions, posterior, hbranch = self.model.forward(x0[z:z + 4, :, :, :], sample_posterior=False)
            if self.args.hbranchz:
                hb_all.append(posterior.sample())
            else:
                hb_all.append(hbranch)
            reconstructions_all.append(reconstructions)

        hbranch = torch.cat(hb_all, dim=0)
        # print("hbranch : ", hbranch.shape)
        reconstructions_all = torch.cat(reconstructions_all, dim=0).detach().to('cpu', non_blocking=True)
        Xup = torch.nn.Upsample(size=(self.upsample_size[0]*8, self.upsample_size[1], self.upsample_size[2]), mode='trilinear')(
            x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
        Xup = Xup[0, 0, ::].permute(2, 0, 1).detach().to('cpu', non_blocking=True)  # .numpy()  # (Z, X, Y))
        del hb_all
        return reconstructions_all, Xup, hbranch

    def get_ae_decode(self, hbranch):
        if self.args.hbranchz:
            hbranch = self.model.decoder.conv_in(hbranch.cuda(0))
        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (C, X, Y, Z)
        # print("hbranch d : ", hbranch.shape)
        if self.gpu:
            hbranch = hbranch.cuda()
            
        out = self.model.net_g(hbranch, method='decode')
        Xout = out['out0'].detach().to('cpu', non_blocking=True)  # (1, C, X, Y, Z)
        XupX = Xout[0, 0, ::].permute(2, 0, 1)  # .numpy()

        if "seg" in self.args.save:
            XupX_seg = Xout[0, 1, ::].permute(2, 0, 1)  # .numpy()
            # XupX_seg = out['out1']
            # XupX_seg = XupX_seg[0, 0, ::].permute(2, 0, 1).detach().cpu()
            return XupX, XupX_seg
        return XupX, "_"




def get_gan_out(args, x0, model):
    #x0 = [x.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2) for x in x0]
    #x0 = torch.cat(x0, dim=1).cuda()  # concatenate all the input channels
    # (Z, C, X, Y)
    gpu = args.gpu
    x0 = x0.permute(1, 2, 3, 0).unsqueeze(0)
    if gpu:
        x0 = x0.cuda()

    if 0:
        z = model(x0, method='encode')[-1].detach().cpu()
        if gpu:
            z = z.cuda()
        XupX = model(z, method='decode')['out0'].detach().cpu()
        XupX = XupX[0, 0, :, :, :]  # (X, Y, Z)

    with torch.cuda.amp.autocast(enabled=args.fp16):
        out = model(x0)
        XupX = out['out0'].detach().cpu()[0, 0, :, :, :]   # out0 for original, out1 for psuedo-mask
    Xup = x0.detach().cpu().squeeze()#.numpy()

    XupX = XupX.permute(2, 0, 1)#torch.transpose(XupX, (2, 0, 1))  # (Z, X, Y)
    Xup = Xup.permute(2, 0, 1)#torch.transpose(Xup, (2, 0, 1))

    if "seg" in args.save:
        XupX_seg = out['out1'].detach().cpu()
        XupX_seg = XupX_seg[0, 0, ::].permute(2, 0, 1)
        return XupX, Xup, XupX_seg
    return XupX, Xup, "_"


def get_ae_out(args, x0, model):
    hbranchz = args.hbranchz
    gpu = args.gpu
    if gpu:
        x0 = x0.cuda()
    hb_all = []

    for z in range(0, 32, 4):
        with torch.cuda.amp.autocast(enabled=args.fp16):
            reconstructions, posterior, hbranch = model.forward(x0[z:z+4, :, :, :], sample_posterior=False)
        if hbranchz:
            hb_all.append(posterior.sample())
        else:
            hb_all.append(hbranch)

    hbranch = torch.cat(hb_all, dim=0).detach().cpu()
    del hb_all

    if hbranchz:
        hbranch = model.decoder.conv_in(hbranch.cuda(0))

    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0) # (C, X, Y, Z)
    if gpu:
        hbranch = hbranch.cuda()

    out = model.net_g(hbranch, method='decode')

    XupX = out['out0']  # (1, C, X, Y, Z)
    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
    Xup = Xup[0, 0, ::].permute(2, 0, 1).detach().cpu()#.numpy()  # (Z, X, Y))
    XupX = XupX[0, 0, ::].permute(2, 0, 1).detach().cpu()#.numpy()
    if "seg" in args.save:
        XupX_seg = out['out0'][0, 1, ::].permute(2, 0, 1).detach().cpu()  # .numpy()
        # XupX_seg = out['out1']
        # XupX_seg = XupX_seg[0, 0, ::].permute(2, 0, 1).detach().cpu()
        return XupX, Xup, XupX_seg
    return XupX, Xup, "_"


def get_ae_out_seperate(args, x0, model, method="forward"):
    hbranchz = args.hbranchz
    gpu = args.gpu
    if gpu:
        x0 = x0.cuda()
    hb_all = []
    for z in range(0, 32, 4):
        with torch.cuda.amp.autocast(enabled=args.fp16):
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