import torch
import torchmetrics
from torchmetrics.image.kid import KID
from torchmetrics.image.fid import FID
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image
import os
import tifffile as tiff
from tqdm import tqdm
#from utils.data.util.data_utils import imagesc
import numpy as np


def load_images_from_folder(folder_path, irange):
    images = []
    for filename in sorted(os.listdir(folder_path))[irange[0]:irange[1]]:
        if filename.endswith((".tif")):
            img_path = os.path.join(folder_path, filename)
            img = tiff.imread(img_path)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
            images.append(img_tensor)
    images = torch.stack(images, 0)
    images = (images + 1) / 2
    images = (images * 255).type(torch.uint8)
    images = images.repeat(1, 3, 1, 1, 1)
    return images.cuda()


def load_images_3d(folder_path, ix):

    file_list = sorted(os.listdir(folder_path))

    img = tiff.imread(os.path.join(folder_path, file_list[ix]))
    img = (img - img.min()) / (img.max() - img.min())
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    img = (img * 255).type(torch.uint8)
    img = img.repeat(1, 3, 1, 1, 1)

    return img


def calculate_metrics(option, folder1, folder2, irange, subset_size=50):
    # Set seed for reproducibility
    torch.manual_seed(123)

    # Load images

    # Initialize metrics
    if option == 'kid':
        metrics = KID(subset_size=subset_size).cuda()
    elif option == 'fid':
        metrics = FID().cuda()
    elif option == 'lpips':
        metrics = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()

    metrics_all = np.array([])
    for ix in tqdm(range(*irange)):

        img1 = load_images_3d(folder1, ix).permute(4, 1, 2, 3, 0).squeeze().cuda()
        img2 = load_images_3d(folder2, ix).permute(4, 1, 2, 3, 0).squeeze().cuda()

        #if option in ['kid', 'fid']:
        # Update metrics with images from both folders
        metrics.update(img1, real=True)
        metrics.update(img2, real=False)

        # Compute metrics
        if option in ['kid']:
            metric_mean, metric_std = metrics.compute()
            #print(f": {metric_mean:.4f} ± {metric_std:.4f}")
        elif option in ['fid']:
            metric_mean = metrics.compute()
            #print(f": {metric_mean:.4f}")

    metrics_all = np.append(metrics_all, metric_mean.detach().cpu().numpy())

    #else:
    #    print(metrics(imgs_dist1, imgs_dist2))
    return metrics_all


if __name__ == '__main__':
    # All: 353: 216
    # Usage

    # ori / linear/ smore / present
    # 0.45 / 0.43/ 0.458/ 0.417
                        #/ 0.305

    # 0.401      0.384   0.406   # 0.373

    #  362                   # 332

    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_paper_figure/'

    folder1 = root + 'original/aval/'
    #folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/zya2d"
    folder2 = root + 'outaval587/xz2d/'
    metrics_all = calculate_metrics('kid', folder1, folder2, irange=(0, 100, 1))

    print((metrics_all.mean(), metrics_all.std()))

    #kid  zx: 0.337, 0.1957  zy:  0.307, 0.208
    #fid  zx: 300, 198  zy:  280, 208
    #print(f"metrics: {metrics_mean:.4f} ± {metrics_std:.4f}")