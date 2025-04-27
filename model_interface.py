import torch

from defenses.jpeg_compression.jpeg_pass import jpeg_pass
from defenses.random_crop.random_crop import random_crop
from defenses.random_noise.random_noise import random_noise


class ModelInterface:
    def __init__(self, models, noise=None, noise_level=0.0, threshold=0.0):
        self.models = models
        self.noise = noise
        self.noise_level = noise_level
        self.threshold = threshold
        self.send_models_to_cuda()

        if self.noise == "jpeg":
            if self.models[0].dataset in ["cifar10", "cifar100"]:
                self.jpeg_compr = jpeg_pass(self.noise_level, 32, 32).cuda()
            else:
                self.jpeg_compr = jpeg_pass(self.noise_level, 224, 224).cuda()

    def __call__(self, images, vanilla=False):
        with torch.no_grad():
            return self.get_probs(images, vanilla)

    def send_models_to_cuda(self):
        for model in self.models:
            model.model = model.model.cuda()

    def get_probs(self, images, vanilla=False):
        with torch.no_grad():
            outs = self.models[0].get_probs(images)

        if self.threshold == 0.0 or vanilla or ("pni" in self.noise or "rse" in self.noise or "at" in self.noise):
            return outs
        else:
            # find maximum
            maxes = torch.max(outs, axis=1)

            # set maximum prediction to zero if beneath a given threshold
            above = maxes[0] > self.threshold

            if (~above).any():
                if self.noise == "rnd":
                    imgs = random_noise(self.noise_level, images[~above])
                elif self.noise == "crop":
                    imgs = random_crop(self.noise_level, images[~above])
                elif self.noise == "jpeg":
                    imgs = (self.jpeg_compr.forward((images[~above] * 255).permute(0, 2, 3, 1)) / 255).permute(0, 3, 1,
                                                                                                               2)

                with torch.no_grad():
                    outs[~above] = self.models[0].get_probs(imgs)

        return outs

    def compute_accuracy(self, loader, vanilla=False):
        with torch.no_grad():
            total = 0
            correct = 0
            for data in loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # calculate outputs by running images through the network
                with torch.no_grad():
                    outputs = self.get_probs(images, vanilla=vanilla)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().cpu().item()
        return correct / total, correct

    def compute_accuracy_fast(self, loader, vanilla=False):
        with torch.no_grad():
            total = 0
            correct = 0
            for data in loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # calculate outputs by running images through the network
                with torch.no_grad():
                    outputs = self.get_probs(images, vanilla=vanilla)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().cpu().item()
                break
        return correct / total ,correct
