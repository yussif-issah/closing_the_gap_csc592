import torch

from attacks.surfree.utils.utils import atleast_kdim


def get_init_with_noise(model, X, y):
    init = X.clone()
    p = model(X).argmax(1)

    # invoke attack with 200 images to make sure we get convergence...
    n_images = int(len(X) / 2)

    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)),
            (X + 0.5 * torch.randn_like(X)).clip(0, 1),
            init)
        p = model(init).argmax(1)

        missclassified_idx = torch.where(p != y)[0]

        if len(missclassified_idx) >= n_images:
            return init[missclassified_idx[0:n_images]], missclassified_idx[0:n_images]

    return init[0:n_images], torch.arange(0, 100, 1)
