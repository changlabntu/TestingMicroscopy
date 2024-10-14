import torch
import torchmetrics
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image
import os
import tifffile as tiff
from tqdm import tqdm


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
    images = images.repeat(1, 3, 1, 1)
    return images.cuda()


def calculate_metrics(option, folder1, folder2, irange, subset_size=50):
    # Set seed for reproducibility
    torch.manual_seed(123)

    # Initialize metrics
    if option == 'kid':
        metrics = KernelInceptionDistance(subset_size=subset_size).cuda()
    elif option == 'fid':
        metrics = FrechetInceptionDistance().cuda()
    elif option == 'lpips':
        metrics = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()

    # Load images
    for ix in tqdm(range(*irange)):
        imgs_dist1 = load_images_from_folder(folder1, irange=(ix, ix+irange[2]))[:, :, :, :]
        imgs_dist2 = load_images_from_folder(folder2, irange=(ix, ix+irange[2]))[:, :, :, :]

        #if option in ['kid', 'fid']:
        # Update metrics with images from both folders
        metrics.update(imgs_dist1, real=True)
        metrics.update(imgs_dist2, real=False)
        # Compute metrics
    try:
        metric_mean, metric_std = metrics.compute()
        print(f": {metric_mean:.4f} ± {metric_std:.4f}")
    except:
        metric_mean = metrics.compute()
        print(f": {metric_mean:.4f}")
    #else:
    #    print(metrics(imgs_dist1, imgs_dist2))
    return metric_mean


if __name__ == '__main__':
    # All: 353: 216
    # Usage
    folder1 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/xya2d"
    #folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/zya2d"
    folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/test/expanded3d/zxa3d"
    metric_mean = calculate_metrics('kid', folder1, folder2, irange=(0, 1840, 512))
    #kid  zx: 0.337, 0.1957  zy:  0.307, 0.208
    #fid  zx: 300, 198  zy:  280, 208
    #print(f"metrics: {metrics_mean:.4f} ± {metrics_std:.4f}")