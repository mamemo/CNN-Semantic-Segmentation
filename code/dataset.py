"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to create the dataset and dataloaders.
"""

import pandas as pd
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as albu


class CustomDataset(Dataset):
    """
        Defines a Custom Dataset
    """
    
    def __init__(self, ids, masks, aug_transf, preproc_transf):
        """
            init Constructor

            @param self Object.
            @param ids Path to the images.
            @param masks Annotations of the images.
            @param transf Transformations to apply.
        """
        super().__init__()

        # Transforms
        self.aug_transf = aug_transf
        self.preproc_transf = preproc_transf

        # Images IDS amd Labels
        self.ids = ids
        self.masks = masks

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
        id_mask = self.masks[index]

        # Open Image
        image = cv2.imread(id_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Open Annotation
        mask = cv2.imread(id_mask, 0)
        mask = np.expand_dims(mask, axis=-1)

        # Applies transformations
        # Apply augmentations
        if self.aug_transf:
            sample = self.aug_transf(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preproc_transf:
            sample = self.preproc_transf(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return self.data_len


def read_dataset(dir_img):
    """
        read_dataset Read the dataset from a csv file.

        @param dir_img Path to the csv file
    """

    images = pd.read_csv(dir_img)
    ids = images['ID_IMG'].tolist()
    masks = images['ANNOTATION'].tolist()
    return ids, masks


def get_training_augmentation(img_size):
    train_transform = [
        albu.Resize(img_size, img_size),

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(img_size, img_size)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_aug_dataloader(train_file, img_size, batch_size, proc_fn):
    """
        get_aug_dataloader Creates and return a dataloader with data augmentation.

        @param train_file Path to the training images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
        @param data_mean Mean values of the dataset (for normalization).
        @param data_std Standard deviation values of the dataset (for normalization).
    """
    
    # Read the dataset
    ids, masks = read_dataset(train_file)

    print("Training Dataset Size: ", len(ids))

    # Create the dataset
    train_dataset = CustomDataset(ids=ids, masks=masks,\
                        aug_transf=get_training_augmentation(img_size=img_size),\
                        preproc_transf=get_preprocessing(proc_fn))

    # Create the loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_dataloader(data_file, img_size, batch_size, proc_fn, data_split = 'Validation'):
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
    ids, masks = read_dataset(data_file)

    print(data_split+" Dataset Size: ", len(ids))

    # Create the dataset
    dataset = CustomDataset(ids=ids, masks=masks,\
                        aug_transf=get_validation_augmentation(img_size=img_size),\
                        preproc_transf=get_preprocessing(proc_fn))

    # Create the loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
