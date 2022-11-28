#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:29:32 2019

@author: tsmotlp
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random
# import cv2
# from skimage.measure import compare_ssim as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import scipy.stats as st
import skimage.color as skcolor
# import os
import torch

torch.cuda.current_device()
import torch.nn as nn
import torchvision.models as torchmodels
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.datasets import MNIST, CIFAR10, LSUN, ImageFolder


class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_color_jittering=False,
            crop_ratio=(0.9, 1.1)
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_color_jittering = with_color_jittering
        self.crop_ratio = crop_ratio

    def transform(self, img):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        return img

    def transform_triplets(self, img, gt1, gt2):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        gt1 = TF.to_pil_image(gt1)
        gt1 = TF.resize(gt1, [self.img_size, self.img_size])

        gt2 = TF.to_pil_image(gt2)
        gt2 = TF.resize(gt2, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)
            gt1 = TF.hflip(gt1)
            gt2 = TF.hflip(gt2)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)
            gt1 = TF.vflip(gt1)
            gt2 = TF.vflip(gt2)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)
            gt1 = TF.rotate(gt1, 90)
            gt2 = TF.rotate(gt2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)
            gt1 = TF.rotate(gt1, 180)
            gt2 = TF.rotate(gt2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)
            gt1 = TF.rotate(gt1, 270)
            gt2 = TF.rotate(gt2, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            gt1 = TF.adjust_hue(gt1, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt1 = TF.adjust_saturation(gt1, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            gt2 = TF.adjust_hue(gt2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt2 = TF.adjust_saturation(gt2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))
            gt1 = TF.resized_crop(
                gt1, i, j, h, w, size=(self.img_size, self.img_size))
            gt2 = TF.resized_crop(
                gt2, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        gt1 = TF.to_tensor(gt1)
        gt2 = TF.to_tensor(gt2)

        return img, gt1, gt2


class CTDataset(Dataset):

    def __init__(self, root_dir, img_size=128, is_train=True):
        self.root_dir = root_dir
        if is_train:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'train_small', '*.png'))
            self.dir_noise = glob.glob(os.path.join(self.root_dir, 'self_ul_new_train2_bright', '*.png'))
        else:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'test_small', '*.png'))
            self.dir_noise = glob.glob(os.path.join(self.root_dir, 'self_ul_new_test2_bright', '*.png'))
        self.lennoise = len(self.dir_noise)
        self.list = list(range(0, self.lennoise))
        self.list.extend(self.list)
        random.shuffle(self.list)
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir)
        # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        dir_noise = self.dir_noise[self.list[idx]]
        img_noise = cv2.imread(dir_noise)

        h1, w1, _ = img_mix.shape
        crop_a = random.randint(0, h1 - self.img_size)
        crop_b = random.randint(0, w1 - self.img_size)
        gt1 = img_mix[crop_a: crop_a + self.img_size, crop_b: crop_b + self.img_size, :]

        h1_n, w1_n, _ = img_noise.shape
        crop_a_n = random.randint(0, h1_n - self.img_size)
        crop_b_n = random.randint(0, w1_n - self.img_size)
        noise_patch = img_noise[crop_a_n: crop_a_n + self.img_size, crop_b_n: crop_b_n + self.img_size, :]

        # noise_patch = img_noise[36: 36 + self.img_size, 36: 36 + self.img_size, :]

        input = gt1 + noise_patch

        # suff = '_' + this_dir.split('_')[-1]
        # this_gt_dir = this_dir.replace('rainy_image', 'ground_truth').replace(suff, '.jpg')
        # gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        # gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        input, gt1, noise_patch = self.augm.transform_triplets(input, gt1, noise_patch)

        data = {
            'input': input,
            'gt1': gt1,
            'noise': noise_patch
        }

        return data


class CTblackDataset(Dataset):

    def __init__(self, root_dir, img_size=128, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        if self.is_train:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'train_small', '*.png'))
            self.dir_noise = glob.glob(os.path.join(self.root_dir, 'self_ul_new_train2_bright', '*.png'))
            self.dir_black = glob.glob(os.path.join(self.root_dir, 'black', '*.png'))
        else:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'test_small', '*.png'))
            self.dir_noise = glob.glob(os.path.join(self.root_dir, 'self_ul_new_test2_bright', '*.png'))
        self.lennoise = len(self.dir_noise)
        self.list = list(range(0, self.lennoise))
        self.list.extend(self.list)
        random.shuffle(self.list)

        self.img_size = img_size
        if self.is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        if self.is_train:
            return len(self.dirs) + 500
        else:
            return len(self.dirs)

    def __getitem__(self, idx):
        if self.is_train:
            if idx < len(self.dirs):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                this_dir = self.dirs[idx]
                img_mix = cv2.imread(this_dir)
                # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

                dir_noise = self.dir_noise[self.list[idx]]
                img_noise = cv2.imread(dir_noise)

                h1, w1, _ = img_mix.shape
                crop_a = random.randint(0, h1 - self.img_size)
                crop_b = random.randint(0, w1 - self.img_size)
                gt1 = img_mix[crop_a: crop_a + self.img_size, crop_b: crop_b + self.img_size, :]

                # noise_patch = img_noise[36: 36 + self.img_size, 36: 36 + self.img_size, :]
                h1_n, w1_n, _ = img_noise.shape
                crop_a_n = random.randint(0, h1_n - self.img_size)
                crop_b_n = random.randint(0, w1_n - self.img_size)
                noise_patch = img_noise[crop_a_n: crop_a_n + self.img_size, crop_b_n: crop_b_n + self.img_size, :]

                input = gt1 + noise_patch

                # suff = '_' + this_dir.split('_')[-1]
                # this_gt_dir = this_dir.replace('rainy_image', 'ground_truth').replace(suff, '.jpg')
                # gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
                # gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

                input, gt1, noise_patch = self.augm.transform_triplets(input, gt1, noise_patch)
            else:
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                black_index = random.randint(0, len(self.dir_black) - 1)
                black_img = cv2.imread(self.dir_black[black_index])

                noise_index = random.randint(0, len(self.dir_black) - 1)
                noise_img = cv2.imread(self.dir_black[noise_index])

                input_index = random.randint(0, len(self.dir_black) - 1)
                input = cv2.imread(self.dir_black[input_index])


                black_img = black_img[36: 36 + self.img_size, 36: 36 + self.img_size, :]
                noise_img = noise_img[36: 36 + self.img_size, 36: 36 + self.img_size, :]
                input = input[36: 36 + self.img_size, 36: 36 + self.img_size, :]
                input, gt1, noise_patch = self.augm.transform_triplets(input, black_img, noise_img)

            data = {
                'input': input,
                'gt1': gt1,
                'noise': noise_patch
            }
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            this_dir = self.dirs[idx]
            img_mix = cv2.imread(this_dir)
            # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

            dir_noise = self.dir_noise[self.list[idx]]
            img_noise = cv2.imread(dir_noise)

            h1, w1, _ = img_mix.shape
            crop_a = random.randint(0, h1 - self.img_size)
            crop_b = random.randint(0, w1 - self.img_size)
            gt1 = img_mix[crop_a: crop_a + self.img_size, crop_b: crop_b + self.img_size, :]

            # noise_patch = img_noise[36: 36 + self.img_size, 36: 36 + self.img_size, :]
            h1_n, w1_n, _ = img_noise.shape
            crop_a_n = random.randint(0, h1_n - self.img_size)
            crop_b_n = random.randint(0, w1_n - self.img_size)
            noise_patch = img_noise[crop_a_n: crop_a_n + self.img_size, crop_b_n: crop_b_n + self.img_size, :]

            input = gt1 + noise_patch

            # suff = '_' + this_dir.split('_')[-1]
            # this_gt_dir = this_dir.replace('rainy_image', 'ground_truth').replace(suff, '.jpg')
            # gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
            # gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

            input, gt1, noise_patch = self.augm.transform_triplets(input, gt1, noise_patch)
            data = {
                'input': input,
                'gt1': gt1,
                'noise': noise_patch
            }

        return data


def get_dataset(root_dir):
    train_dataset = CTblackDataset(root_dir, is_train=True)
    val_dataset = CTblackDataset(root_dir, is_train=False)
    return train_dataset, val_dataset


if __name__ == '__main__':
    train_dataset, val_dataset = get_dataset('/home/ted/ztt/dataset/denoise')
    print(val_dataset[9021])