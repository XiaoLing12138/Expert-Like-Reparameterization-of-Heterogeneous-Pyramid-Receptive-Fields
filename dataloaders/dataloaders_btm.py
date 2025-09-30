import os
import random

import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageFilter
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class BTMDataset(Dataset):
    def __init__(self, root_path, transform, mode):
        super(BTMDataset, self).__init__()
        self.root = root_path
        self.mode = mode

        self.datas = []

        if mode == "test":
            img_list = pd.read_csv(os.path.join(self.root, "Test.csv"))
        elif mode == "train":
            img_list = pd.read_csv(os.path.join(self.root, "Train.csv"))
        else:
            img_list = pd.read_csv(os.path.join(self.root, "Valid.csv"))


        for line in np.array(img_list).tolist():
            self.datas.append([root_path+"/"+line[0][78:], line[1]])

        if mode == "train":
            random.shuffle(self.datas)

        self.transform = transform

    def __getitem__(self, item):
        img_path, label = self.datas[item]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.datas)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class BTMDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.img_resize, scale=(0.2, 1.0)),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                    ),
                    transforms.RandomRotation(45),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_resize, self.img_resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        dataset = BTMDataset(
            root_path=self.root_dir,
            transform=transform,
            mode=mode
        )

        if mode == "train":
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )

        return loader, dataset
