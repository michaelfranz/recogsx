import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Ratio of split to use for validation.
VALID_SPLIT = 0.16  # Around 1/6 (i.e. 1000 out of 6000 images)
# Batch size.
BATCH_SIZE = 32
# Path to data root directory.
ROOT_DIR = '../input/mfc_images'


# Training transforms
def get_train_transform():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # input images preprocessed to all be size 98x12 hence no resize necesssary
    ])
    return train_transform


# Validation transforms
def get_valid_transform():
    return get_train_transform()


# Initial entire datasets,
# same for the entire and test dataset.
def get_datasets():
    dataset = datasets.ImageFolder(ROOT_DIR, transform=get_train_transform())
    dataset_test = datasets.ImageFolder(ROOT_DIR, transform=get_valid_transform())
    print(f"Classes: {dataset.classes}")
    dataset_size = len(dataset)
    print(f"Total number of images: {dataset_size}")

    valid_size = int(VALID_SPLIT*dataset_size)

    # Training and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])

    print(f"Total training images: {len(dataset_train)}")
    print(f"Total valid_images: {len(dataset_valid)}")
    return dataset_train, dataset_valid, dataset.classes


# Training and validation data loaders.
def get_data_loaders():
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, valid_loader, dataset_classes
