import torch
import numpy as np
import torch.nn as nn

import config
import models


class TCS_Net(nn.Module):
    def __init__(self):
        super(TCS_Net, self).__init__()
        self.block_size = config.para.block_size

        self.NUM_UNITS_PATCH = 6
        self.NUM_UNITS_PIXEL = 1

        self.DIM = 32
        self.PIXEL_EMBED = 32
        self.PATCH_DIM = 8  # todo
        self.idx = config.para.block_size // self.DIM

        # pixel-wise
        pixel_embedding = np.random.normal(0.0, (1 / self.PIXEL_EMBED) ** 0.5, size=(1, self.PIXEL_EMBED))
        self.pixel_embedding = nn.Parameter(torch.from_numpy(pixel_embedding).float(), requires_grad=True)
        self.transform_pixel_wise = nn.ModuleList()
        for i in range(self.NUM_UNITS_PIXEL):
            self.transform_pixel_wise.append(models.OneUnit(dim=self.PIXEL_EMBED, dropout=0.5, heads=1))
        pixel_detaching = np.random.normal(0.0, (1 / self.PIXEL_EMBED) ** 0.5, size=(self.PIXEL_EMBED, 1))
        self.pixel_detaching = nn.Parameter(torch.from_numpy(pixel_detaching).float(), requires_grad=True)

        # patch-wise
        self.transform_patch_wise = models.Units(dim=self.PATCH_DIM, depth=self.NUM_UNITS_PATCH, dropout=0.5, heads=8)
        # self.transform_patch_wise = PatchTransform(
        #     dim=self.PATCH_DIM ** 2, num_layers=self.NUM_UNITS_PATCH, drop_out=0.5)

        # sampling and init recon
        points = self.DIM ** 2
        p_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(points, int(config.para.rate * points)))
        self.P = nn.Parameter(torch.from_numpy(p_init).float(), requires_grad=True)
        self.R = nn.Parameter(torch.from_numpy(np.transpose(p_init)).float(), requires_grad=True)

        w_init = 1e-3 * np.ones(self.DIM)
        self.w = nn.Parameter(torch.from_numpy(w_init).float(), requires_grad=True)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        y = self.sampling(inputs)
        recon, tmp = self.recon(y, batch_size)
        return recon, tmp

    def sampling(self, inputs):
        inputs = inputs.to(config.para.device)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.DIM, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.DIM, dim=2), dim=0)
        inputs = inputs.reshape(-1, self.DIM ** 2)
        y = torch.matmul(inputs, self.P.to(config.para.device))
        return y

    def recon(self, y, batch_size):
        # init
        init = torch.matmul(y, self.R.to(config.para.device)).reshape(-1, 1, self.DIM, self.DIM)

        recon = torch.cat(torch.split(init, split_size_or_sections=batch_size * self.idx, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)

        # patch
        recon = self.block2patch(recon, self.PATCH_DIM)
        patch = self.transform_patch_wise(recon)
        recon = recon - patch[0]
        recon = self.patch2block(recon, self.block_size, self.PATCH_DIM)

        recon = torch.cat(torch.split(recon, split_size_or_sections=self.DIM, dim=3), dim=0)
        recon = torch.cat(torch.split(recon, split_size_or_sections=self.DIM, dim=2), dim=0).reshape(-1, self.DIM ** 2, 1)

        # pixel
        recon_pixel = torch.matmul(recon, self.pixel_embedding)
        for i in range(self.NUM_UNITS_PIXEL):
            recon_pixel, _ = self.transform_pixel_wise[i](recon_pixel)
        recon_pixel = torch.matmul(recon_pixel, self.pixel_detaching)
        recon = recon.reshape(-1, 1, self.DIM, self.DIM) - recon_pixel.reshape(-1, 1, self.DIM, self.DIM) * self.w

        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size * self.idx, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)

        return recon, None

    @staticmethod
    def block2patch(inputs, size):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=size, dim=3), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=size, dim=2), dim=1)
        return inputs

    @staticmethod
    def patch2block(inputs, block_size, patch_size, ori_channel=1):
        assert block_size % patch_size == 0, f"block size {block_size} should be divided by patch size {patch_size}."
        idx = int(block_size / patch_size)
        outputs = torch.cat(torch.split(inputs, split_size_or_sections=ori_channel * idx, dim=1), dim=2)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=ori_channel, dim=1), dim=3)
        return outputs
