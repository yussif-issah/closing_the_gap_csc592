import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode

import main


def get_dataset(dataset, split):
    if split == "val":
        train = False
    elif split == "train":
        train = True
    else:
        raise NotImplementedError("Unknown dataset split.")

    if dataset == "cifar10":
        dataset_path = main.get_absolute_path("paths/dataset") + f"/{dataset}"
        data = torchvision.datasets.CIFAR10(root=dataset_path, download=True, train=train,
                                            transform=get_transforms(dataset))
    elif dataset == "cifar100":
        dataset_path = main.get_absolute_path("paths/dataset") + f"/{dataset}"
        data = torchvision.datasets.CIFAR100(root=dataset_path, download=True, train=train,
                                             transform=get_transforms(dataset))
    elif dataset == "imagenet":
        dataset_path = main.get_config_value("dataset/imagenet/path")
        data = torchvision.datasets.ImageNet(root=dataset_path, split=split, transform=get_transforms(dataset))

    return data


def get_transforms(dataset):
    if dataset == "cifar10":
        tts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    elif dataset == "cifar100":
        tts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    elif dataset == "imagenet":
        tts = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=232, interpolation=InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
        ])

    return tts


def get_images(model, dataset, min_confidence, num_images):
    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    chosen_idx = []

    X = []
    Y = []
    i = 0
    while len(X) < num_images:
        elem = dataset.__getitem__(indices[i])
        x = elem[0].unsqueeze(0).cuda()  # .permute(0, 2, 3, 1)
        y = elem[1]

        probs = model(x)
        yhat = probs.argmax(1)
        if y == yhat and probs[0, yhat] >= min_confidence:
            X.append(x)
            Y.append(y)
            chosen_idx.append(indices[i])
        i += 1

    X = torch.cat(X, 0)

    Y = torch.Tensor(Y).int()

    return X, Y
