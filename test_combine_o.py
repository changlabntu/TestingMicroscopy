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

import networks
import models
from utils.data_utils import imagesc
from utils.model_utils import read_json_to_args, import_model, load_pth

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial

"""
1. Inference model
2. Calculate score
3. Save images only
"""

def reverse_log(x):
    return np.power(10, x)

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

def recreate_volume_folder(destination, mc=1):
    # remove and recreate the folder
    if os.path.exists(destination + 'xy/'):
        shutil.rmtree(destination + 'xy/')
    if os.path.exists(destination + 'ori/'):
        shutil.rmtree(destination + 'ori/')
    os.makedirs(destination + 'xy/', exist_ok=True)
    os.makedirs(destination + 'ori/', exist_ok=True)
    if mc > 1:
        os.makedirs(destination + 'xyvar/')

class MicroTest:
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None

    def init_params(self):
        # Init / Update Model Args
        self.args = self.update_args()
        # Init / Update Data Args
        self.config, self.kwargs = self.update_data_args('test/' + self.args.config + '.yaml')
        print("kwargs : ", self.kwargs)

    def update_args(self):
        parser = argparse.ArgumentParser()
        # projects
        parser.add_argument('--config', type=str, default="config", help='which config file')
        parser.add_argument('--option', type=str, default="Default", help='which dataset to use')
        parser.add_argument('--prj', type=str, default="/ae/cut/1/", help='name of the project')
        parser.add_argument('--epoch', type=str, default='3000', help='epoch #')
        parser.add_argument('--mc', type=str, default=1, help='monte carlo inference, mean over N times')
        parser.add_argument('--model_type', type=str, default='AE', help='GAN or AE or Upsample or None')
        parser.add_argument('--assemble', action='store_true', default=False)
        parser.add_argument('--hbranchz', action='store_true', default=False) # hidden branch for AE middle layer
        parser.add_argument('--gpu', action='store_true', default=False)
        parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')
        parser.add_argument('--reverselog', action='store_true', default=False)
        parser.add_argument('--assemble_method', type=str, default='tiff', help='tiff or zarr method while assemble images')

        return parser.parse_args()

    def update_data_args(self, config_name):
        with open(config_name, 'r') as f:
            config = yaml.safe_load(f)

        kwargs = config.get(self.args.option, {})
        if not kwargs:
            raise ValueError(f"Option {self.args.option} not found in the configuration.")
        return config, kwargs

    def update_model(self):
        model = None
        upsample = None
        # get GAN model
        if self.args.model_type == 'GAN':
            model_name = self.config['SOURCE'] + '/logs/' + self.kwargs['dataset'] + self.args.prj + '/checkpoints/net_g_model_epoch_' + str(
                self.args.epoch) + '.pth'
            print(model_name)
            model = torch.load(model_name, map_location=torch.device('cpu'))

            # rergister model method
            from utils.model_utils import get_gan_out
            self.get_model_result = get_gan_out

        # get AE model
        if self.args.model_type == 'AE':
            component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
            root = self.config['SOURCE'] + '/logs/' + self.kwargs['dataset'] + self.args.prj
            args = read_json_to_args(root + '0.json')  # load config json file

            # dynamically load module
            model_module = import_model(root, model_name=args.models)
            model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
            model = load_pth(model, root=root, epoch=self.args.epoch, model_names=component_names)

            # rergister model method
            from utils.model_utils import get_ae_out
            self.get_model_result = get_ae_out

        if self.args.model_type in ['AE', 'GAN', 'Upsample']:
            upsample = torch.nn.Upsample(size=self.kwargs['upsample_params']['size'], mode='trilinear')
            if self.args.gpu:
                model = model.cuda()
                upsample = upsample.cuda()

        if self.args.model_type in ['AE', 'GAN']:
            for param in model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                model = model.half()
        self.model = model
        self.upsample = upsample

    def get_data(self, norm=True):
        # 我覺得這要改，有兩種情況，一她只要開圖存，不見得真的要paired data,二 他要inference
        image_path = [x for x in self.kwargs.get("image_path", [])]  # if image path is a file
        image_list_path = self.kwargs.get("image_list_path")  # if image path is a directory

        x0 = []
        if image_path:
            for i in range(len(image_path)):
                img = tiff.imread(image_path[i])
                if norm:
                    img = self._norm_x0(img, self.kwargs["norm_method"][i],
                                       self.kwargs['exp_trd'][i], self.kwargs['exp_ftr'][i], self.kwargs['trd'][i])
                x0.append(img)

        # I'm not sure image list path is used for 2D images or need deal with many 3D images cube
        # new method with 2D loading images
        # assert 2D image is (X, Y) output will be [(Z, X, Y), (Z, X, Y)]
        elif image_list_path:
            ids = sorted(os.listdir(image_list_path[0]))
            for num, i in enumerate(image_list_path):
                img = np.stack([tiff.imread(os.path.join(i, id)) for id in ids], 0)
                if norm:
                    img = self._norm_x0(img, self.kwargs["norm_method"][num],
                                       self.kwargs['exp_trd'][num], self.kwargs['exp_ftr'][num], self.kwargs['trd'][num])
                x0.append(img)

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

            out_aug = []
            for i, aug in enumerate(input_augmentation):  # (Z, C, X, Y)
                # augmentation
                input = self._test_time_augementation(patch, method=aug)

                # here is the forward
                if self.args.fp16 and self.args.gpu:
                    input = input.half()
                    with torch.cuda.amp.autocast():
                        out, Xup = self.get_model_result(self.args, input, self.model)
                else:
                    out, Xup = self.get_model_result(self.args, input, self.model)  # (Z, C, X, Y)

                # augmentation back
                out = self._test_time_augementation(out.unsqueeze(1), method=aug)
                Xup = self._test_time_augementation(Xup.unsqueeze(1), method=aug)

                # reshape back to 2d for input
                out = out.squeeze()
                Xup = Xup.squeeze()

                out_aug.append(out)

            out_aug = torch.stack(out_aug, 0)
            out = torch.mean(out_aug, 0)

            out_all.append(out.numpy())

        out_all = np.stack(out_all, axis=3).mean(axis=3)

        return out_all, Xup.numpy()

    def test_assemble(self):
        dz, dx, dy = self.kwargs['assemble_params']['dx_shape']
        zrange = range(*self.kwargs['assemble_params']['zrange'])
        xrange = range(*self.kwargs['assemble_params']['xrange'])
        yrange = range(*self.kwargs['assemble_params']['yrange'])

        recreate_volume_folder(destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], 'cycout'), mc=self.kwargs['mc'])
        self._test_over_volumne(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                               destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], 'cycout'),
                               input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:])

    def save_images(self, outpath, img, axis=None):
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
        if self.args.reverselog:
            img = reverse_log(img)

        if axis is not None:
            img = np.transpose(img, axis)

        # 儲存影像
        tiff.imwrite(outpath, img)

    def assemble_microscopy_volumne(self, zrange, xrange, yrange, source):
        """
        先把最後大小算出來，算完之後開個依樣大的final圖
        """
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

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

                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)

                    # load and crop
                    x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
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

        one_stack = np.concatenate(one_stack, axis=0).astype(np.float32)

        tiff.imwrite(source[:-1] + '.tif', one_stack)

    def assemble_microscopy_volume_memmap(self, zrange, xrange, yrange, source, output_path="tmp_xy.tif"):
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        # 計算最終大圖的尺寸
        total_z = len(zrange) * (self.kwargs['assemble_params']['dx_shape'][0] * 8 - 2 * C0 - S0) + S0 # z 8 倍
        total_x = len(xrange) * (self.kwargs['assemble_params']['dx_shape'][1] - 2 * C1 - S1) + S1
        total_y = len(yrange) * (self.kwargs['assemble_params']['dx_shape'][2] - 2 * C2 - S2) + S2

        with tiff.TiffWriter(output_path, bigtiff=True) as tif:
            empty_slice = np.zeros((total_x, total_y), dtype='float32')
            for _ in range(total_z):
                tif.write(empty_slice, contiguous=True, dtype='float32')

        volume = tiff.memmap(output_path, mode='r+', shape=(total_x, total_z, total_y), dtype='float32')
        # volume = tiff.memmap(output_path, mode='r+', shape=(total_z, total_x, total_y), dtype='float32')

        # 創建一個內存映射的大數組
        # volume = np.memmap(output_path, dtype='float32', mode='w+', shape=(total_z, total_x, total_y))
        posi_x, posi_y, posi_z = 0, 0, 0

        for nx, ix in enumerate(xrange):
            posi_z = 0
            for nz, iz in enumerate(zrange):
                posi_y = 0
                for ny, iy in enumerate(yrange):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1
                    # 加載並裁剪小圖塊
                    filename = os.path.join(source, f"{iz}_{ix}_{iy}.tif")
                    x = tiff.imread(filename)
                    stride_x, stride_y, stride_z = x.shape[1] - 2*C0 - S0, x.shape[2] - 2*C1 - S1, x.shape[0] - 2*C2 - S2
                    if C0 > 0:
                        cropped = x[C0:-C0, :, :]
                    if C1 > 0:
                        cropped = cropped[:, C1:-C1, :]
                    if C2 > 0:
                        cropped = cropped[:, :, C2:-C2]

                    # 應用權重
                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                              size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    cropped = np.multiply(cropped, w)
                    cropped = cropped.transpose(1, 0, 2)

                    volume[
                    posi_x:posi_x + cropped.shape[1],
                    posi_z:posi_z + cropped.shape[0],
                    posi_y:posi_y + cropped.shape[2]] += cropped
                    posi_y += stride_y
                posi_z += stride_z
            posi_x += stride_x

        # 確保數據寫入磁盤
        del volume

    def assemble_microscopy_volume_zarr_parallel(self, zrange, xrange, yrange, source,
                                                 output_path="assembled_volume.zarr"):
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        dx_shape = self.kwargs['assemble_params']['dx_shape']

        # 計算最終大圖的尺寸
        total_z = len(zrange) * (dx_shape[0] * 8 - 2 * C0 - S0) + S0
        total_x = len(xrange) * (dx_shape[1] - 2 * C1 - S1) + S1
        total_y = len(yrange) * (dx_shape[2] - 2 * C2 - S2) + S2

        # 創建 zarr 文件，設置適當的分塊大小
        store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=store, overwrite=True)
        volume = root.create_dataset(
            'volume',
            shape=(total_z, total_x, total_y),
            dtype='float32',
            chunks=(256, 256, 256),  # 根據具體情況調整分塊大小
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )

        # 預計算所有權重
        print("預計算權重...")
        weights_cache = self._precompute_weights(zrange, xrange, yrange)

        # 定義處理 Patch 的函數
        def process_patch(iz, ix, iy):
            C0, C1, C2 = self.kwargs['assemble_params']['C']
            S0, S1, S2 = self.kwargs['assemble_params']['S']
            dx_shape = self.kwargs['assemble_params']['dx_shape']

            filename = os.path.join(source, f"{iz}_{ix}_{iy}.tif")
            if not os.path.exists(filename):
                print(f"文件不存在: {filename}")
                return None
            x = tiff.imread(filename)

            # 裁剪邊界
            cropped = x[C0:-C0, C1:-C1, C2:-C2] if C0 > 0 and C1 > 0 and C2 > 0 else x

            # 應用權重
            w = weights_cache.get((iz, ix, iy))
            if w is not None:
                cropped = np.multiply(cropped, w)
            else:
                print(f"權重不存在: iz={iz}, ix={ix}, iy={iy}")

            # 計算位置
            z_idx = zrange.index(iz)
            x_idx = xrange.index(ix)
            y_idx = yrange.index(iy)

            start_z = z_idx * (dx_shape[0] * 8 - 2 * C0 - S0)
            start_x = x_idx * (dx_shape[1] - 2 * C1 - S1)
            start_y = y_idx * (dx_shape[2] - 2 * C2 - S2)

            end_z = start_z + cropped.shape[0]
            end_x = start_x + cropped.shape[1]
            end_y = start_y + cropped.shape[2]

            return (start_z, start_x, start_y, cropped)

        # 使用 ThreadPoolExecutor 進行並行處理
        with ThreadPoolExecutor(max_workers=8) as executor:
            patch_coords = [(iz, ix, iy) for iz in zrange for ix in xrange for iy in yrange]
            for result in tqdm(executor.map(lambda coords: process_patch(*coords), patch_coords),
                               total=len(patch_coords)):
                if result is None:
                    continue
                start_z, start_x, start_y, cropped = result
                end_z = start_z + cropped.shape[0]
                end_x = start_x + cropped.shape[1]
                end_y = start_y + cropped.shape[2]
                # 將處理後的 Patch 添加到大圖中
                volume[start_z:end_z, start_x:end_x, start_y:end_y] += cropped

        print(f"組裝完成，保存至 {output_path}")

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

    def _test_over_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination,
                          input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in xrange:
                for iz in tqdm(zrange):
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        out_all, patch = test_model(x0, input_augmentation)

                        # 處理輸出
                        out = out_all.mean(axis=3).astype(np.float32)
                        if self.args.reverselog:
                            out = reverse_log(out)
                            patch = reverse_log(patch)

                        # 將寫入任務加入隊列
                        write_queue.put((iz, ix, iy, out, patch, out_all.std(axis=3)))
        except Exception as e:
            print(f"Error during processing: {e}")
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

    def _norm_x0(self, x0, norm_method, exp_trd, exp_ftr, trd):
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

    def _do_upsample(self, x0):
        assert self.upsample is not None, "upsample is None call get_model first to update upsample"
        return self.upsample(x0)


if __name__ == "__main__":
    tester = MicroTest()
    # Update model and upsample
    tester.update_model()

    # Here you can register data
    # x0 = tester.get_data()

    # Here you can test model then save it
    # out, patch = tester.test_model(x0, [None, 'transpose', 'flipX', 'flipY'])
    # tester.save_images("tmp.tif", out, (1, 0, 2))

    # Do test assemble then save patch
    # tester.test_assemble()

    # show or save assemble big image
    zrange = range(*tester.kwargs['assemble_params']['zrange'])
    xrange = range(*tester.kwargs['assemble_params']['xrange'])
    yrange = range(*tester.kwargs['assemble_params']['yrange'])
    import time
    start = time.time()
    tester.assemble_microscopy_volume_zarr_parallel(zrange=zrange, xrange=xrange, yrange=yrange,
                                source=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], 'cycout/xy/'))
    end = time.time()
    print("cost : ", end - start)


    # python test_combine_o.py  --prj /ae/cut/1/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --assemble