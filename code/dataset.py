"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to create the dataset and dataloaders.
"""

import pandas as pd
# from PIL import Image
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
# from imgaug import augmenters as iaa
import albumentations as A


class CustomDataset(Dataset):
    """
        Defines a Custom Dataset
    """
    
    def __init__(self, ids, anns, transf):
        """
            init Constructor

            @param self Object.
            @param ids Path to the images.
            @param anns Annotations of the images.
            @param transf Transformations to apply.
        """
        super().__init__()

        # Transforms
        self.transforms = transf

        # Images IDS amd Labels
        self.ids = ids
        self.anns = anns

        # Calculate len of data
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        """
            getitem Method to get one image.

            @param self Object.
            @param index The position of the image in the dataset.
        """
        # Get an ID of a specific image
        id_img = self.ids[index]
        id_ann = self.anns[index]

        # Open Image
        img = cv2.imread(id_img)

        # Open Annotation
        ann = cv2.imread(id_ann, 0)

        # Applies transformations
        augmented = self.transforms(image=img, mask=ann)

        return (id_img, augmented['image'], augmented['mask'])

    def __len__(self):
        return self.data_len


def read_dataset(dir_img):
    """
        read_dataset Read the dataset from a csv file.

        @param dir_img Path to the csv file
    """

    images = pd.read_csv(dir_img)
    ids = images['ID_IMG'].tolist()
    anns = images['ANNOTATION'].tolist()
    return ids, anns


def get_aug_dataloader(train_file, img_size, batch_size, data_mean, data_std):
    """
        get_aug_dataloader Creates and return a dataloader with data augmentation.

        @param train_file Path to the training images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
        @param data_mean Mean values of the dataset (for normalization).
        @param data_std Standard deviation values of the dataset (for normalization).
    """
    
    # Read the dataset
    ids, anns = read_dataset(train_file)

    #Transformations
    train_transform = A.Compose([
        A.Resize(img_size, img_size),

        # Augmentation
        A.OneOrOther(
            A.GaussianBlur(),
            A.IAASharpen(),
            p=0.25
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.8),    
        A.RandomGamma(p=0.8),

        A.Normalize(mean=data_mean, std=data_std),
        A.pytorch.ToTensorV2()
    ])

    print("Training Dataset Size: ", len(ids))

    # Create the dataset
    train_dataset = CustomDataset(ids=ids, anns=anns, transf=train_transform)

    # Create the loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_dataloader(data_file, img_size, batch_size, data_mean, data_std, data_split = 'Validation'):
    """
        get_dataloader Creates and returns a dataloader with no data augmentation.

        @param data_file Path to the images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
        @param data_mean Mean values of the dataset (for normalization).
        @param data_std Standard deviation values of the dataset (for normalization).
        @param data_split Training process where this is dataloader is used (Val or Test).
    """

    # Read the dataset
    ids, anns = read_dataset(data_file)

    # Transformations
    test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=data_mean, std=data_std),
        A.pytorch.ToTensorV2()
    ])

    print(data_split+" Dataset Size: ", len(ids))

    # Create the dataset
    dataset = CustomDataset(ids=ids, anns=anns, transf=test_transform)

    # Create the loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
