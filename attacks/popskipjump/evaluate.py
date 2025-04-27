import torch

from attacks.popskipjump.tracker import DiaryPage, Diary


def compute_distance(x1, x2):
    return torch.norm(x1 - x2)


def classify_images(model_interface, raw):
    NUM_IMAGES = len(raw)
    NUM_ITERATIONS = 32

    D = torch.zeros(size=(NUM_IMAGES, NUM_ITERATIONS)).cuda()

    MODEL_CALLS = torch.zeros(size=(NUM_IMAGES, NUM_ITERATIONS, 1)).cuda()

    correct_predictions_clean = [False] * NUM_IMAGES
    correct_predictions_adv_opposite_defense = [False] * NUM_IMAGES

    ACCS = [[] for i in range(2)]

    originals = []
    advs = []

    for image in range(NUM_IMAGES):
        diary: Diary = raw[image]
        x_star = diary.original.unsqueeze(dim=0)
        label = diary.true_label

        originals.append(x_star)

        # check classification of original image
        correct_predictions_clean[image] = model_interface(x_star, vanilla=True).argmax(dim=1) == label.cpu().item()

        calls = [None] * 1

        if len(diary.iterations) > 0:
            # find the last non-empty iteration -> shouldnt be an issue as only the last iteration is saved
            # only the distances are pre-intialized with zeroes hence we need to find the last non-zero one while
            # plotting..
            page: DiaryPage = diary.iterations[-1]

            # adversary opposite
            adversary = page.opposite.unsqueeze(dim=0)
            advs.append(adversary)

            correct_predictions_adv_opposite_defense[image] = not model_interface(adversary, vanilla=False).argmax(
                dim=1) == label.cpu().item()

            for iteration in range(len(diary.iterations)):
                page: DiaryPage = diary.iterations[iteration]
                D[image, iteration] = compute_distance(x_star, page.opposite)

                calls[0] = page.calls.opposite

                MODEL_CALLS[image, iteration] = torch.tensor(calls)
        else:
            advs.append(torch.zeros_like(x_star))

    ACCS[0] = list(map(bool, correct_predictions_clean))
    ACCS[1] = list(map(bool, correct_predictions_adv_opposite_defense))

    dump = {
        'distance': D,
        'accuracies': ACCS,
        'model_calls': MODEL_CALLS
    }

    return dump, originals, advs
