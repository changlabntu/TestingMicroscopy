import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse
import os, importlib, sys


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


def read_json_to_args(json_file):
    with open(json_file, 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    return args

class ModelProcesser:
    def __init__(self, args, model, upsample = None):
        self.args = args
        self.model = model
        self.upsample = upsample
        self.gpu = args.gpu

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

        if self.args.save_seg:
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

        for z in range(0, 32, 4):
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                reconstructions, posterior, hbranch = self.model.forward(x0[z:z + 4, :, :, :], sample_posterior=False)
            if self.args.hbranchz:
                hb_all.append(posterior.sample())
            else:
                hb_all.append(hbranch)
            reconstructions_all.append(reconstructions)

        hbranch = torch.cat(hb_all, dim=0).detach().cpu()
        reconstructions_all = torch.cat(reconstructions_all, dim=0).detach().cpu()
        Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(
            x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
        Xup = Xup[0, 0, ::].permute(2, 0, 1).detach().cpu()  # .numpy()  # (Z, X, Y))

        del hb_all
        return reconstructions_all, Xup, hbranch

    def get_ae_decode(self, hbranch):
        if self.args.hbranchz:
            hbranch = self.model.decoder.conv_in(hbranch.cuda(0))

        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (C, X, Y, Z)
        if self.gpu:
            hbranch = hbranch.cuda()

        out = self.model.net_g(hbranch, method='decode')

        XupX = out['out0']  # (1, C, X, Y, Z)

        XupX = XupX[0, 0, ::].permute(2, 0, 1).detach().cpu()  # .numpy()
        if self.args.save_seg:
            XupX_seg = out['out0'][0, 1, ::].permute(2, 0, 1).detach().cpu()  # .numpy()
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

    if args.save_seg:
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
    if args.save_seg:
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