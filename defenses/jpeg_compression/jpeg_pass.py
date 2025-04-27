import numpy as np
import torch
import torch.nn as nn


class rgb_to_ycbcr(nn.Module):
    """ Converts Images to a YCbCr tensor
    Input: Tensor(batch,height,width,3)
    Output: Tensor(batch,height,width,3)
     """

    def __init__(self):
        super(rgb_to_ycbcr, self).__init__()
        matrix = torch.tensor(
            [[0.299, -0.168736, 0.5], [0.587, -0.331264, -0.418688],
             [0.114, 0.5, -0.081312]])
        self.matrix = nn.Parameter(matrix)
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))

    def forward(self, dataset):
        images = dataset
        result = torch.tensordot(images, self.matrix, dims=1) + self.shift
        return result


class spatial_filtering(nn.Module):
    """ We filter the images with ratio 4:2:2
      Input: Tensor(batch,height,width,3)
      Output: Tensor(batch,height,width) x Tensor(batch,height,width) x Tensor(batch,height,width)
     """

    def __init__(self):
        super(spatial_filtering, self).__init__()

    def forward(self, images):
        avg_pooling = nn.AvgPool2d(kernel_size=2, stride=(2, 2), count_include_pad=False)
        averaged_cb = avg_pooling(images[:, :, :, 1])
        averaged_cr = avg_pooling(images[:, :, :, 2])
        return images[:, :, :, 0], averaged_cb, averaged_cr


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:  Tensor(batch,height,width)
    Output: Tensor(batch,height*width/64 x h x w)
    """

    def __init__(self):
        super(block_splitting, self).__init__()

    def forward(self, images):
        [batch_size, height, width] = images.size()
        (image_width_fixed, width) = (images, width) if width % 8 == 0 else (
            torch.cat((images,) + (images[:, :, -1].unsqueeze(2),) * (8 - width % 8), dim=2), width + 8 - width % 8)
        (image_height_fixed, height) = (image_width_fixed, height) if height % 8 == 0 else (
            torch.cat((image_width_fixed,) + (image_width_fixed[:, -1, :].unsqueeze(1),) * (8 - height % 8), dim=1),
            height + 8 - height % 8)
        images_reshaped = image_height_fixed.reshape(batch_size, height // 8, 8, width // 8, 8)
        images_transposed = images_reshaped.permute(0, 1, 3, 2, 4)
        result = images_transposed.reshape(batch_size, -1, 8, 8)
        return result


class dct(nn.Module):
    """ Discrete Cosine Transformation
      Input:Tensor(batch,block,height,width)
      Output: Tensor(batch,block,height,width)
      """

    def __init__(self):
        super(dct, self).__init__()
        dct_tensor = torch.zeros((8, 8, 8, 8))
        u = torch.from_numpy(np.arange(8))
        v = torch.from_numpy(np.arange(8))
        for x in range(8):
            for y in range(8):
                dct_tensor[x, y, :, :] = torch.cos((2 * x + 1) * u * np.pi / 16).unsqueeze(1) * torch.cos(
                    (2 * y + 1) * v * np.pi / 16)

        alpha = torch.tensor([1. / np.sqrt(2)] + [1.] * 7)
        self.dct_tensor = nn.Parameter(dct_tensor.float())
        self.scale = nn.Parameter((torch.outer(alpha, alpha) * 0.25).float())

    def forward(self, images):
        images = images - 128
        result = self.scale * torch.tensordot(images, self.dct_tensor, dims=2)
        result.reshape(images.shape)
        return result


class quantization(nn.Module):
    def __init__(self, quality):
        super(quantization, self).__init__()
        quality = 5000 / quality if quality < 50 else 200 - quality * 2
        self.factor = quality / 100

        self.y_table = nn.Parameter(torch.tensor([[16., 12., 14., 14., 18., 24., 49., 72.],
                                                  [11., 12., 13., 17., 22., 35., 64., 92.],
                                                  [10., 14., 16., 22., 37., 55., 78., 95.],
                                                  [16., 19., 24., 29., 56., 64., 87., 98.],
                                                  [24., 26., 40., 51., 68., 81., 103., 112.],
                                                  [40., 58., 57., 87., 109., 104., 121., 100.],
                                                  [51., 60., 69., 80., 103., 113., 120., 103.],
                                                  [61., 55., 56., 62., 77., 92., 101., 99.]]))

        c_table = torch.empty((8, 8))
        c_table.fill_(99)
        c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66],
                                        [24, 26, 56, 99], [47, 66, 99, 99]])
        self.c_table = nn.Parameter(c_table)

    def forward(self, ycbcr):
        y, cb, cr = ycbcr
        y = y.float() / (self.y_table * self.factor)
        y = torch.round(y)

        cb = cb.float() / (self.c_table * self.factor)
        cb = torch.round(cb)

        cr = cr.float() / (self.c_table * self.factor)
        cr = torch.round(cr)
        return y, cb, cr


class jpeg_compression(nn.Module):
    def __init__(self, quality):
        super(jpeg_compression, self).__init__()
        self.layer1 = nn.Sequential(rgb_to_ycbcr(), spatial_filtering())
        self.layer2 = nn.Sequential(block_splitting(), dct())
        self.layer3 = quantization(quality)

    def forward(self, dataset):
        y, cb, cr = self.layer1(dataset)
        y = self.layer2(y)
        cb = self.layer2(cb)
        cr = self.layer2(cr)

        return self.layer3((y, cb, cr))


class ycbcr_to_rgb(nn.Module):
    """ Converts Images to a RGB tensor
    Input : Tensor(batch,height,width,3)
    Output : Tensor(batch,height,width,3)
     """

    def __init__(self):
        super(ycbcr_to_rgb, self).__init__()
        matrix = torch.tensor(
            [[1.000, 1.000, 1.000], [0.000, -0.334, 1.773],
             [1.403, -0.714, 0.000]])
        self.matrix = nn.Parameter(matrix)
        self.shift = nn.Parameter(torch.tensor([0., -128., -128.]))

    def forward(self, images):
        result = torch.tensordot(images + self.shift, self.matrix, dims=1)
        return result


class inverse_spatial_filtering(nn.Module):
    """ Inverses the spatial filtering and merges Y,Cb,Cr
      Input : Tensor(batch,height,width)xTensor(batch,height/2,width/2)xTensor(batch,height/2,width/2)
      Output : Tensor(batch,height,width,3)
     """

    def __init__(self):
        super(inverse_spatial_filtering, self).__init__()

    def forward(self, images):
        y, cb, cr = images
        batch, c_height, c_width = cb.size()
        _, y_height, y_width = y.size()
        new_cb = cb.unsqueeze(3).repeat(1, 1, 2, 2).reshape(batch, c_height * 2, c_width * 2, 1)
        new_cr = cr.unsqueeze(3).repeat(1, 1, 2, 2).reshape(batch, c_height * 2, c_width * 2, 1)
        if y_height != c_height * 2:
            new_cb = torch.cat((new_cb, new_cb[:, -1, :].unsqueeze(1)), dim=1)
            new_cr = torch.cat((new_cr, new_cr[:, -1, :].unsqueeze(1)), dim=1)
        if y_width != c_width * 2:
            new_cb = torch.cat((new_cb, new_cb[:, :, -1].unsqueeze(2)), dim=2)
            new_cr = torch.cat((new_cr, new_cr[:, :, -1].unsqueeze(2)), dim=2)

        return torch.cat((y.unsqueeze(3), new_cb, new_cr), dim=3)


class block_merging(nn.Module):
    """ Merge blocks into image
    Inputs:Tensor(batch,block,height,width) x height x width
    Output:Tensor(batch,height,width)
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, images_and_shape):
        images, height, width = images_and_shape
        new_height = height if height % 8 == 0 else (height + 8 - height % 8)
        new_width = width if width % 8 == 0 else (width + 8 - width % 8)
        batch_size = images.size()[0]
        images_reshaped = images.reshape(batch_size, new_height // 8, new_width // 8, 8, 8)
        images_transposed = images_reshaped.permute(0, 1, 3, 2, 4)
        result = images_transposed.reshape(batch_size, new_height, new_width)
        result = result[:, :height, :width]
        return result


class idct(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:Tensor(batch,block,height,width)
    Output: Tensor(batch,block,height,width)
    """

    def __init__(self):
        super(idct, self).__init__()
        alpha = torch.tensor([1. / np.sqrt(2)] + [1.] * 7)
        self.alpha = nn.Parameter(torch.outer(alpha, alpha).float())
        tensor = torch.zeros((8, 8, 8, 8))
        u = torch.from_numpy(np.arange(8))
        v = torch.from_numpy(np.arange(8))

        for x in range(8):
            for y in range(8):
                tensor[x, y, :, :] = torch.cos((2 * u + 1) * x * np.pi / 16).unsqueeze(1) * torch.cos(
                    (2 * v + 1) * y * np.pi / 16)

        self.tensor = nn.Parameter(tensor.float())

    def forward(self, images):
        images = images * self.alpha
        result = 0.25 * torch.tensordot(images, self.tensor, dims=2) + 128
        result.reshape(images.size())
        return result


class dequantization(nn.Module):
    def __init__(self, quality):
        super(dequantization, self).__init__()

        quality = 5000 / quality if quality < 50 else 200 - quality * 2
        self.factor = quality / 100

        self.y_table = nn.Parameter(torch.tensor([[16., 12., 14., 14., 18., 24., 49., 72.],
                                                  [11., 12., 13., 17., 22., 35., 64., 92.],
                                                  [10., 14., 16., 22., 37., 55., 78., 95.],
                                                  [16., 19., 24., 29., 56., 64., 87., 98.],
                                                  [24., 26., 40., 51., 68., 81., 103., 112.],
                                                  [40., 58., 57., 87., 109., 104., 121., 100.],
                                                  [51., 60., 69., 80., 103., 113., 120., 103.],
                                                  [61., 55., 56., 62., 77., 92., 101., 99.]]))

        c_table = torch.empty((8, 8))
        c_table.fill_(99)
        c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])
        self.c_table = nn.Parameter(c_table)

    def forward(self, ycbcr):
        y, cb, cr = ycbcr
        return y * (self.y_table * self.factor), cb * (self.c_table * self.factor), cr * (self.c_table * self.factor)


class jpeg_decompression(nn.Module):
    def __init__(self, quality, height, width):
        super(jpeg_decompression, self).__init__()

        self.y_height = height
        self.y_width = width

        c_height = int(np.floor((height - 2) / 2 + 1))
        c_width = int(np.floor((width - 2) / 2 + 1))

        self.c_height = c_height
        self.c_width = c_width

        self.layer4 = nn.Sequential(dequantization(quality))
        self.layer3 = idct()
        self.layer2 = block_merging()
        self.layer1 = nn.Sequential(inverse_spatial_filtering(), ycbcr_to_rgb())

    def forward(self, image):
        image = self.layer4(image)

        y, cb, cr = image

        y = self.layer3(y)
        y = self.layer2((y, self.y_height, self.y_width))

        cb = self.layer3(cb)
        cb = self.layer2((cb, self.c_height, self.c_width))

        cr = self.layer3(cr)
        cr = self.layer2((cr, self.c_height, self.c_width))

        image = self.layer1((y, cb, cr))

        image = torch.clamp(image, 0, 255)
        return image


class jpeg_pass(nn.Module):
    def __init__(self, quality, height, width):
        super(jpeg_pass, self).__init__()
        self.c = jpeg_compression(quality)
        self.d = jpeg_decompression(quality, height, width)

    def forward(self, image):
        return self.d(self.c(image))
