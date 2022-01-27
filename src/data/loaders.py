"""Create PyTorch's DataLoaders"""

import logging

# Torch libraries
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Custom libraries
from .datasets import PascalCustomDataset as Dataset
from .datasets import (
    CentralCrop,
    Normalise,
    RandomCrop,
    RandomMirror,
    ResizeScale,
    ToTensor,
)
from .hsi_dataset import HSIDataset


def create_loaders(args):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      meta_train_prct (int) : percentage of meta-train.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.

    If train_list == val_list, then divide train_list into meta-train and meta-val.

    Returns:
      train_loader, val loader, do_search (boolean, train_list == val_list).

    """
    ## Transformations during training ##
    logger = logging.getLogger(__name__)

    assert args.fold >= 1 and args.fold <= 5, "Fold number is invalid!"
    all_txtfiles = [f'{args.root_dir}/partition/P1.txt',
                    f'{args.root_dir}/partition/P2.txt',
                    f'{args.root_dir}/partition/P3.txt',
                    f'{args.root_dir}/partition/P4.txt',
                    f'{args.root_dir}/partition/P5.txt']

    # Setup the five-folds
    train_files = []
    for i in range(5):
        if i == (args.fold - 1):
            test_files = all_txtfiles[i]
        else:
            train_files.append(all_txtfiles[i])

    # Check for cross-contamination
    assert any(elem in train_files for elem in
               test_files) is False, "The train and test sets are contaminated!"

    # Train dataloaders
    full_train_dataset = HSIDataset(args.root_dir, txt_files=train_files)

    num_train = len(full_train_dataset)
    split = int(np.floor(0.1 * num_train))  # 10% for validation
    trainset, val_dataset = random_split(full_train_dataset, [num_train - split, split], generator=torch.Generator().manual_seed(42))

    assert len(trainset) + len(val_dataset) == num_train

    do_search = True
    # Split train into meta-train and meta-val
    n_examples = len(trainset)
    n_train = int(n_examples * args.meta_train_prct / 100.0)
    trainset, valset = random_split(trainset, [n_train, n_examples - n_train])

    logger.info(
        " Created train set = {} examples, val set = {} examples; do_search = {}".format(
            len(trainset), len(valset), do_search
        )
    )
    ## Training and validation loaders ##
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size[0],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader, do_search


def get_training_dataloaders(batch_size, num_workers, root_dir, fold: int =1):
    assert fold >= 1 and fold <= 5, "Fold number is invalid!"
    all_txtfiles = [f'{root_dir}/partition/P1.txt',
                    f'{root_dir}/partition/P2.txt',
                    f'{root_dir}/partition/P3.txt',
                    f'{root_dir}/partition/P4.txt',
                    f'{root_dir}/partition/P5.txt']

    # Setup the five-folds
    train_files = []
    for i in range(5):
        if i == (fold-1):
            test_files = all_txtfiles[i]
        else:
            train_files.append(all_txtfiles[i])

    # Check for cross-contamination
    assert any(elem in train_files for elem in test_files) is False, "The train and test sets are contaminated!"

    # Train dataloaders
    full_train_dataset = HSIDataset(root_dir, txt_files=train_files)

    num_train = len(full_train_dataset)
    split = int(np.floor(0.1 * num_train))  # 10% for validation
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train-split, split], generator=torch.Generator().manual_seed(42))

    assert len(train_dataset) + len(val_dataset) == num_train

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.5 * num_train))

    train_weights_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    )

    train_params_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    )

    # Validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )


    print("Dataset information")
    print("-------------------")
    print(f"Train weights set: {len(train_weights_loader.sampler)}")
    print(f"Train params set: {len(train_params_loader.sampler)}")
    print(f"Val set: {len(val_loader.sampler)}")
    print(f"Train files: {train_files}")
    print("-------------------")

    return train_weights_loader, train_params_loader, val_loader
