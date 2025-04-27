import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from robustbench.utils import load_model
from torchvision import models as pymodels
from torchvision.models import ResNet50_Weights

import main
from models.calibration.temperature_scaling import ModelWithTemperature
from models.cifar10.resnet20_pni.models import noise_resnet20
from models.cifar10.resnet34_oat.oat.resnet_OAT import ResNet34OAT
from models.cifar10.vgg16_rse.models import VGG
from models.densenet import densenet121
from models.resnet import resnet34


class Model:
    def __init__(self, model, noise=None, dataset=None, normalization=None, temperature=1.0):
        self.model = model
        self.noise = noise
        self.dataset = dataset

        self.temperature = temperature
        self.normalization = normalization

    def predict(self, images):
        if self.temperature == 0.0:
            return self.model(self.normalization(images))
        else:
            return self.model(self.normalization(images)) / self.temperature

    def __call__(self, images):
        return self.get_probs(images)

    def get_probs(self, images):
        if type(images) != torch.Tensor:
            images = torch.tensor(images, dtype=torch.float32)
        logits = self.predict(images)

        # Softmax maybe included in model? Make sure it is not :)
        assert torch.sum(logits) != 1.0
        probs = torch.softmax(logits, dim=1)

        return probs


def get_model(args, _lambda=None):
    dataset = args.dataset
    architecture = args.arch
    noise = args.defense

    path = main.get_absolute_path("paths/model")
    path = f"{path}/{dataset}/{architecture}"

    mean = main.get_config_value(f"dataset/{dataset}/models/{architecture}/normalization/mean")
    std = main.get_config_value(f"dataset/{dataset}/models/{architecture}/normalization/std")

    model = None
    if dataset == 'cifar10':
        if architecture == "resnet34_oat":
            encoding_mat = np.load(f"{path}/rand_otho_mat.npy")
            model = ResNet34OAT(use2BN=True, FiLM_in_channels=128, encoding_mat=encoding_mat, _lambda=_lambda)
            model = torch.nn.DataParallel(model)
            ckpt = torch.load(f"{path}/latest.pth", map_location=torch.device("cuda"))
            state_dict = ckpt['model']
            model.load_state_dict(state_dict)
        elif architecture == "densenet121":
            model = densenet121(drop_rate=0)
            state_dict = torch.load(path + '.pt', map_location=torch.device("cuda"))
            model.load_state_dict(state_dict)
        elif architecture == "vgg16_rse":
            model = VGG("VGG16", 0.3)
            model.apply(weights_init)
            state_dict = torch.load(path + '/vgg16_noise_0.3.pth', map_location=torch.device("cuda"))
            model.load_state_dict(state_dict)
        elif architecture == "resnet20_pni":
            model = noise_resnet20()
            state_dict = torch.load(path + '/resnet20_pni.pt', map_location=torch.device("cuda"))
            state_tmp = model.state_dict()
            for key in state_tmp.keys():
                state_tmp[key] = state_dict[f"1.{key}"]
            model.load_state_dict(state_tmp)
        elif architecture == "wresnet-70-16_at":
            model = load_model(model_name='Gowal2020Uncovering',
                               dataset='cifar10',
                               threat_model='L2')

    elif dataset == 'cifar100':
        if architecture == "resnet50":
            model = pymodels.resnet50()

            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 1024)
            model.fc = nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(num_ftrs, 1024),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1024, 512),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 256),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 128),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 100)
            )
            state_dict = torch.load(path + '.pt', map_location=torch.device("cuda"))
            model.load_state_dict(state_dict)

    elif dataset == 'imagenet':
        if architecture == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = pymodels.resnet50(weights=weights)

    if model is not None:
        if "oat" not in architecture:
            temperature = main.get_config_value(f"dataset/{dataset}/models/{architecture}/temperature")
        else:
            temperature = main.get_config_value(f"dataset/{dataset}/models/{architecture}/temperature/{_lambda}")

        if args.no_calibration:
            temperature = 1.0

        return Model(model.cuda().eval(), noise, dataset, torchvision.transforms.Normalize(mean=mean, std=std),
                     temperature)
    else:
        print(f"Unknown {dataset} dataset and {model} model.")
        exit()


def calibrate_model(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    cModel = ModelWithTemperature(model)
    cModel.set_temperature(loader)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
