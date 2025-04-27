import torch


def random_noise(noise_level, im):
    imgs = im + noise_level * torch.randn_like(im)
    imgs = torch.clamp(imgs, 0, 1)
    return imgs
