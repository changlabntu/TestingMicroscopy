import argparse
import glob
import json
import os
import shutil
import sys
import time

import threading
import queue
import ipdb
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
import traceback

import networks
import models
from utils.data_utils import imagesc, DataNormalization
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore', message='TiffPage.*read_bytes.*')

import tifffile
warnings.simplefilter('ignore', tifffile.tifffile.TiffFileError)


def reverse_log(x):
    return np.power(10, x)


def recreate_volume_folder(destination, mc=1, folder=["xy", "ori"]):
    # remove and recreate the folder

    for f in folder:
        if os.path.exists(os.path.join(destination, f)) and f in folder:
            shutil.rmtree(os.path.join(destination, f))
        os.makedirs(os.path.join(destination, f), exist_ok=True)

def writer_thread_func(write_queue, destination, args):
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break  # 終止信號

        mode = item[0]
        try:
            if mode == "decode":
                # 從 test_model 來的結果
                _, iz, ix, iy, out_all_mean, out_all_std, out_seg_all = item
                # if args.reverselog:
                #     out_all_mean = reverse_log(out_all_mean)
                if "xy" in args.save:
                    tiff.imwrite(os.path.join(destination, "xy", f"{iz}_{ix}_{iy}.tif"), out_all_mean)
                if "seg" in args.save:
                    tiff.imwrite(os.path.join(destination, "seg", f"{iz}_{ix}_{iy}.tif"), out_seg_all)
                if int(args.mc) > 1:
                    tiff.imwrite(os.path.join(destination, "xyvar", f"{iz}_{ix}_{iy}.tif"), out_all_std)
            elif mode == "full":
                # 從 test_model 來的結果
                _, iz, ix, iy, out_all_mean, patch = item #, out_all_std, out_seg_all = item

                # if args.reverselog:
                #     out_all_mean = reverse_log(out_all_mean)
                #     patch = reverse_log(patch)

                out_all_mean = out_all_mean.mean(axis=-1)
                #out_all_mean = np.permute(out_all_mean, (1, ))
                # (Z, C, X, Y)
                out_all_mean = np.transpose(out_all_mean, (1, 0, 2, 3))
                patch = np.transpose(patch, (1, 0, 2, 3))

                # TEMP: (-1 ~ 1 ) to (0 255) of unit8
                out_all_mean = ((out_all_mean + 1) / 2 * 255).astype(np.uint8)  # (-1, 1) -> (0, 255)
                patch = ((patch + 1) / 2 * 255).astype(np.uint8)

                if "xy" in args.save:
                    tiff.imwrite(os.path.join(destination, "xy", f"{iz}_{ix}_{iy}.tif"), out_all_mean)
                if "ori" in args.save:
                    tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), patch)
                #if "seg" in args.save:
                #    tiff.imwrite(os.path.join(destination, "seg", f"{iz}_{ix}_{iy}.tif"), out_seg_all)
                #if int(args.mc) > 1:
                #    tiff.imwrite(os.path.join(destination, "xyvar", f"{iz}_{ix}_{iy}.tif"), out_all_std)

            elif mode == "encode":
                # 從 test_ae_encode 來的結果
                _, iz, ix, iy, reconstructions, ori, hbranch = item

                # if args.reverselog:
                #     reconstructions = reverse_log(reconstructions)
                #     ori = reverse_log(ori)
                if "recon" in args.save:
                    tiff.imwrite(os.path.join(destination, "recon", f"{iz}_{ix}_{iy}.tif"), reconstructions)
                if "ori" in args.save:
                    tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), ori)
                np.save(os.path.join(destination, "hbranch", f"{iz}_{ix}_{iy}.npy"), hbranch)
            else:
                print(f"未知的模式: {mode}")

        except Exception as e:
            print(f"Error writing files (mode={mode}): {e}")
        finally:
            write_queue.task_done()


class MicroTest:
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None
        self.save_image_datatype = self.args.image_datatype # uint8 # float32 # uint16
        self.normalization = DataNormalization(backward_type=self.save_image_datatype)

    def process_config(self, config, option):
        return {**config['DEFAULT'], **config[option]}

    def init_params(self):
        # Init / Update Model Args
        self.args = self.update_args()
        # Init / Update Data Args
        #self.kwargs, self.kwargs = self.update_data_args('test/' + self.args.config + '.yaml')

        self.kwargs = self.process_config(yaml.safe_load(open('test/' + self.args.config + '.yaml')), self.args.option)

        #print("kwargs : ", self.kwargs)`
        #print("data : ", self.kwargs)

    def update_args(self):
        parser = argparse.ArgumentParser()
        # projects
        parser.add_argument('--config', type=str, default="dpmfull", help='which config file')
        parser.add_argument('--option', type=str, default="VMAT", help='which dataset to use')
        parser.add_argument('--mc', type=str, default=1, help='monte carlo inference, mean over N times')
        parser.add_argument('--testpatch', action='store_true', default=False)
        parser.add_argument('--testcube', action='store_true', default=False)
        parser.add_argument('--gpu', action='store_true', default=False)
        parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')
        #parser.add_argument('--assemble', action='store_true', default=False)
        #parser.add_argument('--assemble_method', type=str, default='tiff',
        #                    help='tiff or zarr method while assemble images')
        parser.add_argument('--save', nargs='+', choices=['ori', 'recon', 'xy'], required=False, help="assign image to save: --save ori recon")
        parser.add_argument('--image_datatype', type=str, default="float32")
        parser.add_argument('--augmentation', type=str, default="encode")
        #parser.add_argument('--roi', type=str, default='')
        parser.add_argument('--reslice', action='store_true', default=False)
        parser.add_argument('--host', type=str, default='dummy')
        parser.add_argument('--port', type=str, default='dummy')

        return parser.parse_args()

    def update_data_args(self, config_name):
        with open(config_name, 'r') as f:
            config = yaml.safe_load(f)

        kwargs = config#.get(self.args.option, {})
        data = config.get(self.args.option, {})
        #if not kwargs:
        #    raise ValueError(f"Option {self.args.option} not found in the configuration.")
        return kwargs, data

    def update_model(self):
        model = None
        upsample = None
        # get GAN model
        if self.kwargs['model_type'] == 'GAN':
            model_name = self.kwargs['SOURCE'] + '/logs/' + self.kwargs['prj'] + '/checkpoints/net_g_model_epoch_' + str(
                self.kwargs['epoch']) + '.pth'
            print(model_name)
            model = torch.load(model_name, map_location=torch.device('cpu'))

        # get AE model
        if self.kwargs['model_type'] == 'AE':
            component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
            root = self.kwargs['SOURCE'] + '/logs/' + self.kwargs['prj']
            args = read_json_to_args(root + '0.json')  # load config json file

            # dynamically load module
            model_module = import_model(root, model_name=args.models)
            model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
            model = load_pth(model, root=root, epoch=self.kwargs['epoch'], model_names=component_names)

        if self.kwargs['model_type'] in ['AE', 'GAN', 'Upsample']:
            upsample = torch.nn.Upsample(size=self.kwargs['upsample_params']['size'], mode='trilinear')
            if self.args.gpu:
                model = model.cuda()
                upsample = upsample.cuda()

        if self.kwargs['model_type'] in ['AE', 'GAN']:
            for param in model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                model = model.half()
        self.model = model
        self.upsample = upsample
        self.model_processer = ModelProcesser(self.args, self.kwargs, self.model, self.kwargs['upsample_params']['size'])

    def get_data(self, norm=True):
        # 我覺得這要改，有兩種情況，一她只要開圖存，不見得真的要paired data,二 他要inference

        #kwargs = self.kwargs

        image_list_path = self.kwargs.get("image_list_path")  # if image path is a directory
        hbranch_path = self.kwargs.get("hbranch_path")

        x0 = []

        if self.kwargs.get("image_path"):
            image_path = [self.kwargs.get("root_path") + x for x in
                          self.kwargs.get("image_path", [])]  # if image path is a file
            print("image_path : ", image_path)
            for i in range(len(image_path)):

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='TiffPage.*read_bytes.*')
                    # or warnings.filterwarnings('ignore', category=TypeError)
                    img = tiff.imread(image_path[i])

                # catch and ignore error
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][i],
                                       self.kwargs['trd'][i])
                x0.append(img)

        # I'm not sure image list path is used for 2D images or need deal with many 3D images cube
        # new method with 2D loading images
        # assert 2D image is (X, Y) output will be [(Z, X, Y), (Z, X, Y)]
        elif self.kwargs.get("image_list_path"):
            image_list_path = [self.kwargs.get("root_path") + x for x in
                               self.kwargs.get("image_list_path")]  # if image path is a directory
            for num, i in enumerate(image_list_path):
                ids = sorted(os.listdir(image_list_path[num]))
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='TiffPage.*read_bytes.*')
                    img = np.stack([tiff.imread(os.path.join(i, id)) for id in ids], 0)
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][num],
                                        self.kwargs['trd'][num])
                x0.append(img)

        elif hbranch_path:
            class HbranchDataset(Dataset):
                def __init__(self, folder):
                    # 搜尋資料夾下所有 npy 檔案，並根據檔名排序（你也可以根據需要修改排序規則）
                    self.files = sorted(glob.glob(os.path.join(folder, "*.npy")))
                    if len(self.files) == 0:
                        raise ValueError(f"No .npy files found in {folder}")

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    # 載入 npy 檔案，可以視需要加入資料轉換（例如轉成 torch.Tensor）
                    return np.load(self.files[idx], allow_pickle=True), self.files[idx]

            hbranch_dataset = HbranchDataset(hbranch_path)
            # 這裡預設 batch_size 為 1，你可以根據需求調整 batch_size 與 shuffle 設定
            x0 = DataLoader(hbranch_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

        else:
            raise ValueError("No valid image path provided.")

        return x0

    def test_model(self, x0, input_augmentation=[None]):
        assert self.model is not None, "model is None call get_model first to update model"

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.args.gpu)  # 初始化 scaler
        out_all = []

        for m in range(self.args.mc):
            d0 = self.kwargs['patch_range']['d0']
            dx = self.kwargs['patch_range']['dx']

            patch = [x[:, :, d0[0]:d0[0] + dx[0], d0[1]:d0[1] + dx[1], d0[2]:d0[2] + dx[2]] for x in x0]
            patch = torch.cat([self._do_upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

            if self.args.fp16 and self.args.gpu:
                input = patch.half()
                with torch.cuda.amp.autocast():
                    out, Xup = self.model_processer.get_model_result(input, input_augmentation)
            else:
                out, Xup = self.model_processer.get_model_result(patch, input_augmentation)

            out_all.append(out.numpy())

        out_all = np.stack(out_all, axis=-1)

        return out_all, Xup.numpy()

    def test_assemble(self, x0, mode="full", input_augmentation=[None, 'transpose', 'flipX', 'flipY']):
        dz, dx, dy = self.kwargs['assemble_params']['dx_shape']

        zrange = tester.kwargs['assemble_params']['zrange']
        yrange = tester.kwargs['assemble_params']['yrange']
        xrange = tester.kwargs['assemble_params']['xrange']

        zrange = range(*[eval(str(x)) for x in zrange])
        xrange = range(*[eval(str(x)) for x in xrange])
        yrange = range(*[eval(str(x)) for x in yrange])

        if mode == "full":
            self._test_over_volumne(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                   destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                                   input_augmentation=input_augmentation)


    def save_images(self, outpath, img, axis=None, norm_method=None, trd=None):
        """
        儲存影像的函數，接受完整的輸出路徑。

        參數：
        - outpath (str): 輸出的檔案路徑，可以包含目錄（例如 "abc/def/xxx.tif"）或僅為檔案名稱（例如 "xxx.tif"）。
        - img (np.ndarray): 要儲存的影像數據。
        - axis (tuple, optional): 影像轉置的軸。
        """
        directory = os.path.dirname(outpath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 處理影像
        # if self.args.reverselog:
        #     img = reverse_log(img)

        if norm_method:
            img = self.normalization.backward_normalization(img, norm_method, trd)

        if axis is not None:
            img = np.transpose(img, axis)

        # 儲存影像
        tiff.imwrite(self.kwargs['root_path'] + outpath, img)

    def _test_time_augementation(self, x, method):
        axis_mapping_func = {"Z": 0, "X": 2, "Y": 3}
        # x shape: (Z, C, X, Y)
        if method == None:
            return x
        elif method.startswith('flip'):
            x = torch.flip(x, dims=[axis_mapping_func[method[-1]]])
            return x
        elif method == 'transpose':
            x = x.permute(0, 1, 3, 2)
            return x

    def _test_over_ae_enc_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in xrange:
                for iz in zrange:
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        reconstructions, ori, hbranch = self.test_ae_encode(x0)

                        reconstructions = self.normalization.backward_normalization(reconstructions,
                                                                            self.kwargs["norm_method"][0],
                                                                            self.kwargs['trd'][0])
                        ori = self.normalization.backward_normalization(ori,
                                                                            self.kwargs["norm_method"][0],
                                                                            self.kwargs['trd'][0])

                        # 將寫入任務加入隊列 "ae",
                        write_queue.put(("encode", iz, ix, iy, reconstructions, ori, hbranch))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _test_over_ae_dec_volumne(self, x0, destination, input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        hbranch_iter = iter(x0)
        try:
            for _ in tqdm(range((len(hbranch_iter)))):
                batch = next(hbranch_iter)
    
                hbranch_data, filename = batch
                iz, ix, iy = filename[0].replace(".npy", "").split("/")[-1].split("_")
                if not torch.is_tensor(hbranch_data):
                    hbranch_data = torch.from_numpy(hbranch_data[0])
                else:
                    hbranch_data = hbranch_data[0]
    
                out_all, out_seg_all = self.test_ae_decode(hbranch_data, input_augmentation)
                out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=3),
                                                                         self.kwargs["norm_method"][0],
                                                                         self.kwargs['trd'][0])
                out_all_std = self.normalization.backward_normalization(out_all.std(axis=3),
                                                                        self.kwargs["norm_method"][0],
                                                                        self.kwargs['trd'][0])
                # out_seg_all = self.normalization.backward_normalization(out_seg_all,
                #                                                         self.kwargs["norm_method"][0],
                #                                                         [0, 255])
                # 將寫入任務加入隊列
                write_queue.put(("decode", iz, ix, iy, out_all_mean, out_all_std, out_seg_all))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _test_over_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination,
                           input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in tqdm(xrange):
                for iz in zrange:
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        out_all, patch = self.test_model(x0, input_augmentation)

                        # SKIP NORMALIZATION
                        if 0:
                            out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=-1),
                                                                                     None,#self.kwargs["norm_method"][0],
                                                                                     self.kwargs['trd'][0])
                            patch = self.normalization.backward_normalization(patch,
                                                                              None,#self.kwargs["norm_method"][0],
                                                                              self.kwargs['trd'][0])

                        # 將寫入任務加入隊列
                        write_queue.put(("full", iz, ix, iy, out_all, patch))#, out_all_std, out_seg_all))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _precompute_weights(self, zrange, xrange, yrange):
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        weights_cache = {}
        for iz in zrange:
            for ix in xrange:
                for iy in yrange:
                    nz = zrange.index(iz)
                    nx = xrange.index(ix)
                    ny = yrange.index(iy)
                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                              size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    weights_cache[(iz, ix, iy)] = w
        return weights_cache

    def _do_upsample(self, x0):
        assert self.upsample is not None, "upsample is None call get_model first to update upsample"
        return self.upsample(x0)


if __name__ == "__main__":
    tester = MicroTest()
    # Update model and upsample
    tester.update_model()
    # Here you can register data

    if tester.args.testpatch or tester.args.testcube or tester.args.reslice:

        x0 = tester.get_data()
        if tester.kwargs.get("norm_mean_std"):
            x0[0] = x0[0] - x0[0].mean()
            x0[0] = x0[0] / x0[0].std()
            x0[0] = x0[0] * tester.kwargs.get("norm_mean_std")[1]
            x0[0] = x0[0] + tester.kwargs.get("norm_mean_std")[0]
        print('Volume shape:  ', print(x0[0].shape))
        print('Volume mean and std', x0[0].mean(), x0[0].std())

    if tester.args.testpatch:
        # 1. Here you can test model with single path image then save it
        tini = time.time()
        out, patch = tester.test_model(x0, [None, 'transpose', 'flipX', 'flipY'][:])
        #
        print(out.shape, patch.shape)
        tester.save_images("out.tif", out.mean(axis=-1), (1, 2, 0, 3), norm_method=None,
                             trd=tester.kwargs['trd'][0]) # norm_method, exp_trd, trd # (Z, C, X, Y, N)
        tester.save_images("patch.tif", patch, (1, 2, 0, 3), norm_method=None,
                                 trd=tester.kwargs['trd'][0])  # (Z, C, X, Y)
        print("Single patch testing time : ", time.time()-tini)

    if tester.args.testcube:

        recreate_volume_folder(
            destination=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"]),
            mc=tester.args.mc,
            folder=["xy", "ori"])

        # save tester.kwargs to yaml file
        with open(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], 'config.yaml'), 'w') as f:
            yaml.dump(tester.kwargs, f)

        zrange = tester.kwargs['assemble_params']['zrange']
        yrange = tester.kwargs['assemble_params']['yrange']
        xrange = tester.kwargs['assemble_params']['xrange']
        zrange = range(*[eval(str(x)) for x in zrange])
        xrange = range(*[eval(str(x)) for x in xrange])
        yrange = range(*[eval(str(x)) for x in yrange])
        tester.test_assemble(x0, mode="full", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:])





