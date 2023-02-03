import os
import cv2
import time
import copy
import torch
import random
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data

import config


class TrainDataPackage:
    r"""
    Packaged images dataset from BSD500/train and BSD500/test (total 400 images) to *.pth,
    We also use VerticalFlip, HorizontalFlip and random generate function from one image (*.jpg) to 150 patches (tensor)
    to enhance our dataset.
    """

    def __init__(self, root="./dataset", transform=None, packaged=True):
        self.training_file = "train.pt"
        self.aug = DataAugment(debug=True)
        self.packaged = packaged
        self.root = root
        self.num = 50  # 50
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ])

        if not (os.path.exists(os.path.join(self.root, self.training_file))):
            print("No packaged dataset file (*.pt) in dataset/, now generating...")
            self.generate()

        if packaged:
            self.train_data = torch.load(os.path.join(self.root, self.training_file))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        img = self.train_data[index]
        return img

    def generate(self):
        paths = [
            os.path.join(self.root, "BSD500/"),
            os.path.join(self.root, "VOCdevkit/VOC2012/JPEGImages/"),
        ]
        patches_list = []

        start = time.time()
        for path in paths:
            for roots, dirs, files in os.walk(path):
                if roots == path:
                    print("Image number: {}".format(files.__len__()))
                    for file in tqdm(files):
                        if file.split('.')[1] == "jpg" or "png" or "tif" or "bmp":
                            temp = os.path.join(path, file)
                            tqdm.write("\r=> Processing " + temp)
                            image = cv2.imread(temp)
                            patches = self.random_patch(image, self.num)
                            patches_list.extend(patches)
        print("Total patches: {}".format(patches_list.__len__()))
        print("Now Packaging...")
        with open(os.path.join(self.root, "train.pt"), 'wb') as f:
            torch.save(patches_list, f)
        end = time.time()
        print("Successfully packaged!, used time: {:.3f}".format(end - start))

    def random_patch(self, image, num):
        size = config.para.block_size
        image = np.array(image, dtype=np.float32) / 255.
        h, w = image.shape[0], image.shape[1]
        if h <= config.para.block_size or w <= config.para.block_size:
            return []
        patches = []
        for n in range(num):
            max_h = random.randint(0, h - size)
            max_w = random.randint(0, w - size)
            patch = image[max_h:max_h + size, max_w:max_w + size, :]
            patch = self.transform(patch)
            patches.append(patch)
        return patches


def train_loader():
    train_dataset = TrainDataPackage()
    dst = torch.utils.data.DataLoader(train_dataset, batch_size=config.para.batch_size, drop_last=True, shuffle=True,
                                      pin_memory=True, num_workers=8, prefetch_factor=8)
    return dst
