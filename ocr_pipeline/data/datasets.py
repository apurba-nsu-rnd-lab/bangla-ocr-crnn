import glob
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils import data

# Albumentations noise
data_transform = A.Compose([
    
    # Pixel level noise
    A.OneOf([
        # GaussNoise -> Random grainy noise
        A.GaussNoise (var_limit=(120.0, 135.0), mean=0, per_channel=True, always_apply=False, p=0.33),
        # PixelDropout -> Salt-and-pepper noise
        A.PixelDropout (dropout_prob=0.005, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.33),
        # ElasticTransform -> used for Pencilic effect
        A.ElasticTransform (alpha=0.15, sigma=0, alpha_affine=0, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.33),
       ], p=0.3),
    
    # Geometric transformations
    A.OneOf([
        # Perspective -> Change the viewing perspective
        A.Perspective (scale=(0.05, 0.05), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.5),
        # Rotate -> Randomly rotate the image up to the limit
        A.Rotate (limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
       ], p=0.3),
    
    # Brightness, Contrast and Blur
    A.OneOf([
        # RandomBrightnessContrast -> Randomly increases and decreases the brightness of the picture up to limit
        A.RandomBrightnessContrast (brightness_limit=[-0.5, 0.5], contrast_limit=[-0.5, 0.5], brightness_by_max=True, always_apply=False, p=0.5),
        # Motion Blur -> Self Explanatory
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
       ], p=0.2),
    
    ToTensorV2(),
    ])


class MJSynthDataset(data.Dataset):
    def __init__(self, img_paths, transform=False):

        self.inp_h = 32
        self.inp_w = 100
        self.img_paths = img_paths
		
        self.transform = data_transform if transform else None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        # print(img_name)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, img_name, idx
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, idx


class SynthDataset(data.Dataset):
    def __init__(self, img_dir, transform=False):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.img_names = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))
		
        self.transform = data_transform if transform else None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, idx

        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, idx

    
class BNHTRDataset(data.Dataset):
    def __init__(self, img_dir, csv_file, transform=False):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.images = pd.read_csv(os.path.join(self.img_dir, csv_file))
        
        self.image_paths = glob.glob(os.path.join(self.img_dir, "Dataset", "*/Words/*/*.[jJ|pP][pP|nN][gG]"))
        self.name_to_path_dict = {image_path.rsplit('/', 1)[-1].split('.', 1)[0].strip():image_path for image_path in self.image_paths}

        self.transform = data_transform if transform else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.name_to_path_dict[self.images.iloc[idx]['Path']]

        # label = self.images.iloc[idx]['Word']
        aid = self.images.iloc[idx]['id']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)
        
        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, aid
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, aid


class BanglaWritingDataset(data.Dataset):
    def __init__(self, img_dir, labels_file, transform=False):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        
        with open(labels_file, 'r') as file:
            mappings = [line.strip() for line in file]

        self.labels_dict = {filename: label for filename, label in (mapping.split(maxsplit=1) for mapping in mappings)}
        self.img_names = sorted(self.labels_dict.keys(), key=lambda x: int(x.split('.')[0]))

        self.transform = data_transform if transform else None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, idx
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, idx 


class DADataset(data.Dataset):
    def __init__(self, img_dir, target_img_dir, csv_file, transform=False):

        self.img_dir = img_dir
        self.target_img_dir = target_img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.img_names = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))
        
        self.tar_images = pd.read_csv(os.path.join(self.target_img_dir, csv_file))
        self.tar_image_paths = glob.glob(os.path.join(self.target_img_dir, "Dataset", "*/Words/*/*.[jJ|pP][pP|nN][gG]"))
        self.name_to_path_dict = {image_path.rsplit('/', 1)[-1].split('.', 1)[0].strip():image_path for image_path in self.tar_image_paths}
        self.t_len = len(self.tar_images)
		
        self.transform = data_transform if transform else None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        # Target Image
        t_idx = idx % self.t_len
        tar_img_path = self.name_to_path_dict[self.tar_images.iloc[t_idx]['Path']]

        # label = self.images.iloc[t_idx]['Word']
        # aid = self.tar_images.iloc[t_idx]['id']
        tar_img = cv2.imread(tar_img_path)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = tar_img.shape
        tar_img = cv2.resize(tar_img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(tar_img.shape)
        tar_img = np.reshape(tar_img, (self.inp_h, self.inp_w, 1))
        #print(tar_img.shape)
        
        if self.transform is not None:
            image = self.transform(image = image)["image"] 
            tar_img = self.transform(image = tar_img)["image"]
            return image, idx, tar_img

        image = image.transpose(2, 0, 1)
        #print(image.shape)
        tar_img = tar_img.transpose(2, 0, 1)
        #print(tar_img.shape)
                
        return image, idx, tar_img
