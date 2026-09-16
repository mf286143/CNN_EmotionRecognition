# prepare dataset, split dataset for training and validation
# create dataloaders (dl) for training, validation, test

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# create preprocessing pipeline for training with augment and evaluation without augmentation
def create_tf(image_size: int):
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),  # dataset in greyscale by default, but loader still loaded it as RGB, set to 1 greyscale channel
        transforms.Resize((image_size, image_size)),    # failsave, dataset in 48x48 resolution by default
        transforms.RandomHorizontalFlip(p=0.5),         # augmentation - 50% to flip image horisontally
        transforms.RandomRotation(10),                  # augmentation - rotate image by random valye between (-10 deg, 10 deg), value chsen to simulate natural sway/tilting of the head
        transforms.ToTensor(),                          # converting to "workable" format for pytorch
    ])


    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.Resize((image_size, image_size)),    # no image augmentation, to keep evaluation consistent
        transforms.ToTensor(),
    ])

    return tf_train, tf_eval

# create dataloaders for training validaiton and test
def create_dl(
    data_path: str,
    batch_size: int,
    val_split: float,
    image_size: int,
    seed: int,
):
    data_path = Path(data_path)
    train_location = data_path / "train"
    test_location = data_path / "test"

    tf_train, tf_eval = create_tf(image_size)

    ds_train_aug = datasets.ImageFolder(
        train_location,
        transform = tf_train,
    )

    ds_train_eval = datasets.ImageFolder(
        train_location,
        transform = tf_eval,
    )

    ds_test = datasets.ImageFolder(
        test_location,
        transform = tf_eval,
    )

    # shuffle and separating by indexes for train and validation sets (test folder was already created by dataset creators)
    generator = torch.Generator().manual_seed(seed)

    indexes = torch.randperm(
        len(ds_train_aug),
        generator = generator,
    ).tolist()

    val_size = int(len(indexes) * val_split)

    val_indexes = indexes[:val_size]
    train_indexes = indexes[val_size:]


    train_subset = Subset(ds_train_aug, train_indexes)
    val_subset = Subset(ds_train_eval, val_indexes)

    # create batches train, validation, test
    dl_train = DataLoader(
        train_subset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )

    dl_val = DataLoader(
        val_subset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    return dl_train, dl_val, dl_test, ds_train_aug.classes