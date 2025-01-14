import torch
import tifffile as tiff
import glob, os
import numpy as np
from tqdm import tqdm
from utils.data_utils import imagesc



root = '/media/ghc/Ghc_data3/BRC/aisr/aisr122424/X2446FG102MM_ExT2_2x/'
destination = 'X4/'
upsample = torch.nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

os.makedirs(root + destination, exist_ok=True)

xlist = sorted(glob.glob(root + 'CamB_1_Transform/*.tif'))

if 0:
    for x in tqdm(xlist[:]):
        img = tiff.imread(x)
        img = img / 1
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img = upsample(img).squeeze().squeeze().numpy()
        img = img.astype(np.uint16)
        tiff.imsave(root + destination + os.path.basename(x), img)


roi = ((565, 856), (774*4, 774*4+1024), (1197*4, 1197*4+1024))

#roi = ((800, 1000), (1200*4, 1200*4+1024), (1100*4, 1100*4+1024))

stack = []
for z in range(roi[0][0], roi[0][1]):
    img = tiff.imread(xlist[z])
    img = img[roi[1][0]:roi[1][1], roi[2][0]:roi[2][1]]
    stack.append(img)
stack = np.stack(stack, 0)
