import os
import cv2
import random

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class DRDataset(Dataset):
    def __init__(self, root_path, transform, mode):
        super(DRDataset, self).__init__()
        self.root = root_path
        self.mode = mode

        self.datas = []

        if mode == "test":
            img_list_1 = pd.read_csv(os.path.join(self.root, "aptos2019/Test.csv"))
        elif mode == "valid":
            img_list_1 = pd.read_csv(os.path.join(self.root, "aptos2019/Valid.csv"))
        elif mode == "train":
            img_list_1 = pd.read_csv(os.path.join(self.root, "aptos2019/Train.csv"))

        for line in np.array(img_list_1).tolist():
            self.datas.append([root_path+"/"+line[0][36:], line[1]])

        if mode != "test":
            random.shuffle(self.datas)

        self.transform = transform

    def __getitem__(self, item):
        img_path, label = self.datas[item]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.datas)


class DRDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=256, root_dir=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if mode == "train":
            transform = transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.RandomAffine(
                    degrees=(-180, 180),
                    scale=(0.8889, 1.0),
                    shear=(-36, 36)),
                transforms.CenterCrop(self.img_resize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                    transforms.Resize((self.img_resize, self.img_resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

        dataset = DRDataset(
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
