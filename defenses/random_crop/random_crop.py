import torch
import torch.nn.functional as F


def random_crop(noise_level, batch):
    permute_back = False
    if batch.ndim == 4 and batch.shape[1] == 3:
        batch = batch.permute(0, 2, 3, 1)
        permute_back = True

    size = batch.shape[1]

    x_start = torch.randint(low=0, high=size + 1 - int(noise_level), size=(1, len(batch)))[0]
    x_end = x_start + int(noise_level)

    y_start = torch.randint(low=0, high=size + 1 - int(noise_level), size=(1, len(batch)))[0]
    y_end = y_start + int(noise_level)

    cropped = [b[x_start[i]:x_end[i], y_start[i]:y_end[i]] for i, b in enumerate(batch)]
    cropped_batch = torch.stack(cropped)

    if cropped_batch.ndim == 4:
        resized = F.interpolate(cropped_batch.permute(0, 3, 1, 2), size, mode='bilinear')
        resized = resized.permute(0, 2, 3, 1)
    else:
        resized = F.interpolate(cropped_batch.unsqueeze(dim=1), size, mode='bilinear')
        resized = resized.squeeze(dim=1)

    if permute_back:
        resized = resized.permute(0, 3, 1, 2)

    return resized
