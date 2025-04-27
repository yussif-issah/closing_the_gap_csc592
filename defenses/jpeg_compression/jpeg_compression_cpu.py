from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


def jpeg_compression(quality, im):
    with torch.no_grad():
        for idx, img in enumerate(im):
            assert torch.is_tensor(img)

            img = ToPILImage()(img)
            savepath = BytesIO()
            img.save(savepath, 'JPEG', quality=int(quality))
            img = Image.open(savepath)

            im[idx] = ToTensor()(img)

    return im
