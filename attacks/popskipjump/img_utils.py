def find_adversarial_images(images, labels):
    ii, ll = images, labels
    cand_img, cand_lbl = [], []
    for i, l in enumerate(ll):
        if l != ll[0]:
            cand_img = [ii[0], ii[i]]
            cand_lbl = [ll[0], ll[i]]
    starts = []
    targeted_labels = []
    for l in labels:
        if l != cand_lbl[0]:
            starts.append(cand_img[0])
            targeted_labels.append(cand_lbl[0])
        else:
            starts.append(cand_img[1])
            targeted_labels.append(cand_lbl[1])
    return starts, targeted_labels
