import torch
import tifffile as tiff
from utils.get_args import get_args
#from models.autoencoder_edit import AutoencoderKL
from utils.data_utils import imagesc
import os, glob
import yaml
import argparse, json
from utils.model_utils import read_json_to_args, load_pth, import_model
import numpy as np
import time
import shutil
import importlib


def test_womac4ae():  # testing the knee ae model
    #prj ='/Fly0B/ae/ae0discstart0/'
    #model_names = ['encoder', 'decoder', 'post_quant_conv', 'quant_conv']

    prj = '/womac4/ae/cyc0/'
    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    epoch = str(60)

    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    #args.ldmyaml = 'ldmaex2x2'

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)

    gan = load_pth(gan, root=root, epoch=epoch, model_names=model_names)

    #img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    #oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    oriX = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/9074878_01.tif')
    #oriX[oriX>= 800] = 800
    oriX = oriX / oriX.max()
    oriX = (oriX - 0.5) / 0.5
    oriX = torch.from_numpy(oriX).unsqueeze(1).float()

    xx = oriX
    reconstructions, posterior, hbranch = gan.forward(xx[:32, :, :, :], sample_posterior=False)
    #imagesc(oriX[0, 0, :, :, 10])
    #imagesc(reconstructions[10, 0, :, :].detach())
    # hbranch (1, 256, 8, 8)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)
    XupX = gan.net_g(hbranch, method='decode')['out0']

    Xup = torch.nn.Upsample(size=(384, 384, 184), mode='trilinear')(xx[:32, :,:,:].permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()
    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()

    imagesc(Xup[200, :, :])
    imagesc(XupX[200, :, :])

    tiff.imwrite('Xup.tif', Xup)
    tiff.imwrite('XupX.tif', XupX)


def test_womac4_vae():  # testing the knee vae
    from models.vae import GAN as VAE
    args = read_json_to_args('/media/ExtHDD01/logs/womac4/vae/0/0.json')
    vae = VAE(args, train_loader=None, eval_loader=None, checkpoints=None)
    net = torch.load('/media/ExtHDD01/logs/womac4/vae/0/checkpoints/generator_model_epoch_500.pth', map_location=torch.device('cpu'))
    vae.generator = net.eval()
    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
    x = tiff.imread(img_list[41])
    x[x >= 800] = 800
    x = x / x.max()
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).unsqueeze(4).float()
    recon, mu, logvar = vae(x)
    imagesc(recon.squeeze().detach().numpy())


def test_Fly0B_gan():  # testing the old Fly0B gan model
    #gan = torch.load('/media/ExtHDD01/logs/Fly0B/IsoScopeXXcut/ngf32lb10/checkpoints/net_g_model_epoch_2000.pth',
    #                 map_location=torch.device('cpu'))

    gan = torch.load('/media/ExtHDD01/logs/Fly0B/IsoScopeXY/ngf32lb10skip4/checkpoints/net_g_model_epoch_2800.pth',
                     map_location=torch.device('cpu'))

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[32:64]], 2)).unsqueeze(0).unsqueeze(1).float()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

    xx = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))

    out = gan(xx)['out0']

    out = out.permute(3, 0, 1, 4, 2).squeeze().detach().numpy()
    tiff.imwrite('XupXgan.tif', out)


def test_Fly0B(prj, epoch):  # testing Fly0B ae model

    prj = '/ae/iso0_ldmaex2x2_lb10/'
    dataset = '/Fly0B/'
    epoch = 500
    path_source = '/media/ExtHDD01/'

    component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    root = path_source + '/logs/' + dataset + prj
    args = read_json_to_args(root + '0.json')  # load config json file

    # dynamically load module
    model_module = import_model(root, model_name=args.models)
    model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
    model = load_pth(model, root=root, epoch=epoch, model_names=component_names).cuda()

    for param in model.parameters():
        param.requires_grad = False

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/Fly0B/train/xyori0828/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[64:96]], 2)).unsqueeze(0).unsqueeze(1).float()#.cuda()

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0].cuda()

    hb_all = []
    for z in range(0, 32, 4):
        reconstructions, posterior, hbranch = model.forward(xx[z:z+4, :, :, :], sample_posterior=False)
        hb_all.append(hbranch)
    hbranch = torch.cat(hb_all, dim=0).detach().cpu()
    del hb_all

    # hbranch (1, 256, 8, 8)
    hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)

    XupX = model.net_g(hbranch.cuda(), method='decode')['out0']

    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze().detach().cpu().numpy()
    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze().detach().cpu().numpy()

    imagesc(Xup[200, :, :])
    imagesc(XupX[200, :, :])

    tiff.imwrite('Xup.tif', Xup)
    tiff.imwrite('XupX.tif', XupX)


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

def z_rescale(xx, trd=6):
    xx=np.log10(xx+1);xx=np.divide((xx-xx.mean()), xx.std());
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd;
    return xx


def test_DPM4Xtc(prj, epoch, irange, hbranchz=False, tc=True, masking=False, input_augmentation=None):

    if input_augmentation == None:
        input_augmentation = [None]

    model_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
    root = '/media/ExtHDD01/logs/' + prj
    args = read_json_to_args(root + '0.json')

    GAN = getattr(__import__('models.' + args.models), args.models).GAN
    gan = GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
    gan = load_pth(gan, root=root, epoch=str(epoch), model_names=model_names)
    gan = gan.cuda()

    for param in gan.parameters():
        param.requires_grad = False

    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/oripatch/*'))
    oriX = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[irange[0]:irange[1]]], 2)).unsqueeze(0).unsqueeze(1).float()
    img_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/DPM4X/ft0patch/*'))
    oriF = torch.from_numpy(np.stack([tiff.imread(y) for y in img_list[irange[0]:irange[1]]], 2)).unsqueeze(0).unsqueeze(1).float()

    tiff.imwrite('XX.tif', oriX.squeeze().numpy())
    tiff.imwrite('FF.tif', oriF.squeeze().numpy())

    xx = oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
    ff = oriF.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

    xx = torch.cat([xx, ff], dim=1).cuda()

    # forward
    for i, aug in enumerate(input_augmentation):
        print(aug)
        input = xx[:32, :, :, :] # (Z, C, X, Y)

        # reshape to 3d for augmentation
        input = input.permute(1, 2, 3, 0).unsqueeze(0) # (1, C, X, Y, Z)

        # augmentation
        input = test_time_augementation(input, method=aug)

        # reshape back to 2d for input
        input = input.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

        hb_all = []
        for z in range(0, 32, 4):
            reconstructions, posterior, hbranch = gan.forward(input[z:z + 4, :, :, :], sample_posterior=False)
            if hbranchz:
                hb_all.append(posterior.sample())
            else:
                hb_all.append(hbranch)

        hbranch = torch.cat(hb_all, dim=0).detach().cpu()
        del hb_all

        #print(hbranch.shape)
        if hbranchz:
            hbranch = gan.decoder.conv_in(hbranch.cuda())

        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)
        #print(hbranch.shape)

        output = gan.net_g(hbranch.cuda(), method='decode')['out0'].detach().cpu()

        # augmentation back
        output = test_time_augementation(output, method=aug)

        if i == 0:
            XupX = output
        else:
            XupX = XupX + output

    XupX = XupX / len(input_augmentation)

    XupX = XupX.permute(3, 0, 1, 4, 2).squeeze(1).detach().numpy()


    Xup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(xx.permute(1, 2, 3, 0).unsqueeze(0))
    fup = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')(ff.permute(1, 2, 3, 0).unsqueeze(0))

    Xup = Xup.permute(3, 0, 1, 4, 2).squeeze(1).cpu().detach().numpy()
    fup = fup.permute(3, 0, 1, 4, 2).squeeze(1).cpu().detach().numpy()

    # printing side view
    if tc:
        to_print = [Xup[:, 0, :, :], XupX[:, 0, :, :]]#[Xup[:, 0, :, :], XupX[:, 0, :, :], Xup[:, 1, :, :], XupX[:, 1, :, :]]
    else:
        to_print = [Xup[:, 0, :, :], XupX[:, 0, :, :], fup[:, 0, :, :]]

    # normalize to 0-1
    to_print = [(x - x.min()) for x in to_print]
    to_print = [(x / x.max()) for x in to_print]

    # normalize by mean and std
    to_print = [(x - x.mean()) for x in to_print]
    to_print = [(x / x.std()) for x in to_print]

    tiff.imwrite('out.tif', np.concatenate(to_print, 2))


if __name__ == '__main__':
    #test_womac4ae()
    #test_Fly0B(prj='/Fly0B/ae/iso0_ldmaex2x2_lb10/', epoch=500)

    destination = '/media/ExtHDD01/logs/DPM4X/ae/iso0_ldmaex2_lb10/'

    #test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10/', epoch=1000, irange=(3968, 3968+32), tc=False, masking=False)
    #test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc/', epoch=1500,
    #         irange=(3968, 3968+32), input_augmentation=[None, 'transpose', 'flip2', 'flip3'][:])
    #test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc_oril1/', epoch=1000, irange=(3968, 3968+32))
    test_DPM4Xtc(prj='/DPM4X/ae/iso0_ldmaex2_lb10_tc_oril1_hbranchz/', epoch=1500, irange=(3968, 3968+32),
                 hbranchz=True, input_augmentation=[None, 'transpose', 'flip2', 'flip3'])
    #test_DPM4Xtc(prj='/DPM4X/ae/2d/3_l1max_advmax/', epoch=500, irange=(3968, 3968+32), hbranchz=True, tc=False)

