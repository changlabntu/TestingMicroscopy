from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss


class PatchSampleF3D(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF3D, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feature_shapes):
        for mlp_id, feat in enumerate(feature_shapes):
            input_nc = feat
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            # mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        # print(len(feats))
        # print([x.shape for x in feats])
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # (B, C, H, W, Z)
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat = feat.permute(0, 2, 3, 4, 1)  # (B, H*W*Z, C)
            feat_reshape = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2] * feat.shape[3], feat.shape[4])
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # (random order of range(H*W))
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device) # first N patches
                    # patch_id = torch.from_numpy(patch_id).type(torch.long).to(feat.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)  # Channel (1, 128, 256, 256, 256) > (256, 256, 256, 256, 256)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        # print([x.shape for x in return_feats]) # (B * num_patches, 256) * level of features
        return return_feats, return_ids
