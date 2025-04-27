import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

import dataset_helper
import model_factory
from attacks.popskipjump.defaultparams import DefaultParams
from attacks.popskipjump.evaluate import classify_images
from attacks.popskipjump.img_utils import find_adversarial_images
from attacks.popskipjump.popskip import PopSkipJump
from attacks.popskipjump.psja_model_interface import PsjaModelInterface
from attacks.surfree.surfree import SurFree
from model_interface import ModelInterface
from torchvision import transforms
to_tensor = transforms.ToTensor()
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", "-a", default=None, help="Attack to be executed.")
    parser.add_argument("--output_folder", "-o", default="output", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=10, help="N images attacks")
    parser.add_argument("--seed", "-seed", type=int, default=2023, help="seed")
    parser.add_argument("--dataset", "-d", default=None, help="cifar10/cifar100/imagenet")
    parser.add_argument("--arch", default=None, help="densenet121/resnet50")
    parser.add_argument("--defense", "-D", default=None, help="Specify defense: rnd, crop, jpeg")
    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="(Optional) Specify threshold used for confidence thresholding.")
    parser.add_argument("--noise_level", type=float, default=100, help="(Optional) Noise level nu.")
    parser.add_argument("--config_path", default="configs")

    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="(Optional) Specify a minimum confidence for the images to attack.")

    parser.add_argument("-l", type=float, default=None)
    parser.add_argument("--epsilon", "-eps", type=float, default=None)
    parser.add_argument("--query_budget", "-q", type=int, default=10000)

    parser.add_argument("--task_id", "-tid", type=int, default=0)

    parser.add_argument("--mode", "-m", default=None)

    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument('--override', action=argparse.BooleanOptionalAction)

    parser.add_argument('--no_calibration', action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def validate_args(args):
    assert args.epsilon is not None and args.epsilon > 0
    assert args.mode in ["attack", "evaluate"]
    assert (args.defense in ["rnd", "crop", "jpeg"]) or "_" in args.arch or "_rse" in args.arch or "_at" in args.arch
    assert args.dataset in get_config_value("dataset").keys()
    assert args.arch in get_config_value(f"dataset/{args.dataset}/models").keys()
    assert args.attack in ["popskipjump", "surfree"]

    if "_pni" in args.arch or "_rse" in args.arch or "_at" in args.arch:
        assert args.dataset == "cifar10"
        assert args.threshold == 0.0
        args.defense = args.arch
    else:
        if args.attack == "surfree":
            assert args.defense == "jpeg"
        elif args.attack == "popskipjump":
            assert args.defense in ["rnd", "crop"]

        if "_oat" in args.arch:
            # use threshold as lambda storage and make sure no thresholding is applied
            args.l = args.threshold
            args.threshold = 0.0

    assert args.epsilon > 0.0

    assert args.query_budget > 0

    if "oat" in args.arch:
        assert 0.0 <= args.l <= 1.0
    else:
        assert args.l is None

    if args.defense == "rnd":
        assert 0.0 <= args.noise_level <= 0.1
    elif args.defense == "crop":
        if args.dataset == "imagenet":
            assert 0 < args.noise_level <= 224
        else:
            assert 0 < args.noise_level <= 32
    elif args.defense == "jpeg":
        assert 0 <= args.noise_level <= 100

    assert 0.0 <= args.threshold <= 1.0
    assert args.output_folder is not None
    assert args.n_images > 0


def fix_randomness(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_images(output_folder, original_images, adversarial_images, valid_advs):
    for idx in valid_advs:
        o = original_images[idx].cpu().squeeze()
        if o.shape[0] != 3:
            o = o.permute(2, 0, 1)

        o = np.array(o * 255).astype(np.uint8)
        img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
        img_o.save(f"{output_folder}/images/{idx}_original.jpg")

        adv_i = adversarial_images[idx].cpu().squeeze()
        if adv_i.shape[0] != 3:
            adv_i = adv_i.permute(2, 0, 1)
        adv_i = np.array(adv_i * 255).astype(np.uint8)
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
        img_adv_i.save(f"{output_folder}/images/{idx}_adversarial.jpg")


def main():
    args = get_args()

    ## VALIDATING ARGS ##
    validate_args(args)

    ## SET GPU to GPU 0 -- Visibility is handled by outer scope ##
    torch.cuda.set_device(0)

    ## SEED RANDOMNESS ##
    fix_randomness(args.seed)

    torch.backends.cudnn.benchmark = True

    ## DETERMINE PATHS ##

    # get path from outer scope
    output_folder = f"{args.output_folder}"

    # CHECK FOR EXISTING PATHS
    if not os.path.exists(output_folder + "/images"):
        os.makedirs(output_folder + "/images", exist_ok=True)

    # SET UP LOGGING

    logFormatter = logging.Formatter(f"%(asctime)s [{args.task_id}]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(output_folder, "output"), mode='w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stderr)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # overriding the print command everywhere to make sure it ends up in file
    print = logging.getLogger().warning

    # OTHER
    print(f"Args: {args}")
    if args.debug:
        print("### DEBUG MODE ENABLED -- NO ACCURACY CHECKS IN PLACE ###")

    if args.no_calibration:
        print("### NO CALIBRATION ###")

    ## LOAD CONFIG ##
    print("Load Main Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path + "/main.json"):
            raise ValueError("{} doesn't exist.".format(args.config_path + "/main.json"))

    torch.no_grad()

    ## LOAD MODEL ##
    print("Load Model")
    model = model_factory.get_model(args, _lambda=args.l)

    ## LOAD DATASET ##
    print("Load Data")
    dataset = dataset_helper.get_dataset(args.dataset, split="val")

    labels = dataset.classes

    X, Y = dataset_helper.get_images(model, dataset, args.min_confidence, args.n_images)
    X = X.cuda()
    Y = Y.cuda()

    # CALIBRATE MODEL
    if "oat" not in args.arch:
        temperature = get_config_value(f"dataset/{args.dataset}/models/{args.arch}/temperature")
    else:
        temperature = get_config_value(f"dataset/{args.dataset}/models/{args.arch}/temperature/{args.l}")

    if temperature == 0.0:
        print("Calibrating model...")
        model_factory.calibrate_model(model, dataset)
        print("Make sure to fill in the temperature value in the config file!")
        exit(0)

    # CHECK FOR PATHS II
    if args.mode == "attack":
        if os.path.exists(output_folder + "/raw_data.pkl"):
            if args.override or args.debug:
                print("Raw data already exists. Overriding.")
            else:
                print("Raw data already exists. Exit.")
                exit(0)
    elif args.mode == "evaluate":
        if os.path.exists(output_folder + "/output.json"):
            if args.override or args.debug:
                print("Parsed data already exists. Overriding.")
            else:
                print("Parsed data already exists. Exit.")
                exit(0)
        elif not os.path.exists(output_folder + "/raw_data.pkl"):
            print("No raw data available for evaluation. Exit.")
            exit(0)

    ## CREATE INTERFACE TO APPLY DEFENSES
    model_interface = ModelInterface([model], noise=args.defense, noise_level=args.noise_level,
                                     threshold=args.threshold)
    if args.mode == "attack":
        acc = 0.0
        ## COMPUTE MAIN TASK ACCURACY ##
        # batch_size might be adapted to available GPU memory
        print("Verifying model accuracy...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False)

        if not args.debug:

            if args.dataset == "imagenet":
                acc,correctWithout = model_interface.compute_accuracy_fast(loader, vanilla=True)
                expected_acc = get_config_value(f"dataset/{args.dataset}/models/{args.arch}/fast_accuracy")
            else:
                acc,correctWithout = model_interface.compute_accuracy(loader, vanilla=True)
                expected_acc = get_config_value(f"dataset/{args.dataset}/models/{args.arch}/accuracy")
                

            ## SANITY CHECK
            print(f"Main acc without defense: {acc}")
            if "oat" in args.arch:
                assert acc - 0.005 <= float(expected_acc[f"{args.l}"]) <= acc + 0.005
            else:
                assert acc - 0.005 <= float(expected_acc) <= acc + 0.005

            ## COMPUTE MAIN TASK ACCURACY WITH DEFENSE ##
            if "_pni" in args.arch or "_rse" in args.arch or "_at" in args.arch:
                pass
            else:
                acc,correctWith = model_interface.compute_accuracy(loader, vanilla=False)
                print(f"Main acc with defense: {acc}")

        print("Attack!")
        time_start = time.time()

        if args.attack == "popskipjump":
            params = DefaultParams()
            params.max_queries = args.query_budget
            attack = PopSkipJump(model_interface, params)
            imgs, labels = X.permute(0, 2, 3, 1), Y

            starts, targeted_labels = find_adversarial_images(imgs, labels)

            median_distance, additional = attack.attack(imgs, labels, starts, targeted_labels,
                                                        iterations=params.num_iterations)
            torch.save([acc, additional], open(f"{output_folder}/raw_data.pkl", 'wb'))

        elif args.attack == "surfree":
            attack_config_path = f"{args.config_path}/{args.attack}.json"

            if attack_config_path is not None:
                if not os.path.exists(attack_config_path):
                    raise ValueError("{} doesn't exist.".format(attack_config_path))

                attack_config = json.load(open(attack_config_path, "r"))

            ## RUN ATTACK ##
            attack_config["init"].update({"max_queries": args.query_budget})
            f_attack = SurFree(**attack_config["init"])
            advs, X, Y = f_attack(model_interface, X, Y, **attack_config["run"])

            torch.save([acc, X, Y, advs, f_attack.get_nqueries()], open(f"{output_folder}/raw_data.pkl", 'wb'))

        print("{:.2f} s to run".format(time.time() - time_start))

    if args.mode == "attack" or args.mode == "evaluate":
        print("Parse!")
        if args.attack == "surfree":

            fix_randomness(args.seed)
            dump = torch.load(open(f"{output_folder}/raw_data.pkl", 'rb'))

            acc, X, Y, advs, nqueries = dump

            ## GATHER RESULTS ##
            print("Results")
            labels_advs = model_interface(advs).argmax(1)
            advs_l2 = (X - advs).norm(dim=[1, 2]).norm(dim=1)
            print(f"nqueries: {[int(nqueries[i].cpu()) for i in nqueries]}")
            print(f"l2dist: {list(advs_l2.cpu().numpy())}")

            filtered_asr = 0
            valid_advs = []

            for image_i in range(len(X)):
                print("Adversarial Image {}:".format(image_i))
                label_o = int(Y[image_i])
                label_adv = int(labels_advs[image_i])

                if label_o != label_adv:
                    valid_advs.append(image_i)

                    if advs_l2[image_i] < args.epsilon:
                        filtered_asr += 1

                print("\t- Original label: {}".format(labels[label_o]))
                print("\t- Adversarial label: {}".format(labels[label_adv]))
                print("\t- l2 = {}".format(advs_l2[image_i]))
                print("\t- {} queries\n".format(nqueries[image_i]))

            asr = len(valid_advs)

            outstuff = {"query": [int(nqueries[i].cpu()) for i in range(len(nqueries))],
                        "l2_dist": [advs_l2[i].cpu().item() for i in range(len(advs_l2))], "parameters": str(args),
                        "asr": f"{asr / len(X)}", "filtered_asr": f"{filtered_asr / len(X)}", "clean_acc": acc}

            with open(os.path.join(output_folder, "output.json"), "w") as f:
                f.write(json.dumps(outstuff))

        elif args.attack == "popskipjump":
            fix_randomness(args.seed)

            raw = torch.load(open(f"{output_folder}/raw_data.pkl", 'rb'))

            acc = raw[0]
            psja_model_interface = PsjaModelInterface(model_interface)

            dump, originals, advs = classify_images(psja_model_interface, raw[1])

            # originals is a list of tensors, each of shape (1, C, H, W)
            # advs   is a list of tensors, same shape

            # 3) Build a proper batch for the clean images
            #    We want an (N, C, H, W) tensor
            batch_clean: torch.Tensor = torch.cat(originals, dim=0)  
            # If originals[i].shape == (1, C, H, W), then cat → (N, C, H, W)

            # 4) Move to the same device as your model
            batch_clean = batch_clean.permute(0, 3, 1, 2)  
            # transforms.Normalize applies channel‐wise over each sample in the batch

            # 6) Predict
            clean_preds = model(batch_clean)
            max_preds_clean = clean_preds.cpu().detach().numpy()
            max_preds_clean = [ np.max(pred) for pred in max_preds_clean]
            
            # 7) (Optional) do the same for the adversarial batch
            batch_adv = torch.cat(advs, dim=0)
            batch_adv = batch_adv.permute(0, 3, 1, 2)  
            adv_preds  = model(batch_adv)
            adv_preds = adv_preds.cpu().detach().numpy()
            max_adv_pred = [np.max(pred) for pred in adv_preds]

            plot(max_preds_clean,max_adv_pred)

def compute_roc_youden(clean_conf, adv_conf, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr = []  # true positive rate: adv_conf < thr
    fpr = []  # false positive rate: clean_conf < thr
    for thr in thresholds:
        tpr.append(np.mean(adv_conf < thr))
        fpr.append(np.mean(clean_conf < thr))
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    return thresholds, tpr, fpr, youden, best_threshold

def plot(clean_conf, adv_conf):
    thresholds, tpr, fpr, youden, best_thr = compute_roc_youden(clean_conf, adv_conf)
    plt.figure()
    plt.plot(thresholds, youden, label="Youden's J")
    plt.axvline(best_thr, color='r', linestyle='--',
            label=f'Optimal τ = {best_thr:.3f}')
    plt.xlabel("Threshold (τ)")
    plt.ylabel("Youden's Index (TPR − FPR)")
    plt.title("Youden's Index vs. Threshold")
    plt.legend()
    plt.show()
    print(f"Optimal threshold (Youden's index): {best_thr:.3f}")
def get_config_value(path):
    dictionary = json.load(open("configs/main.json", "r"))

    for item in path.split("/"):
        dictionary = dictionary[item]

    return dictionary


def get_absolute_path(path):
    file_path = os.path.dirname(os.path.abspath(__file__))
    return file_path + "/" + get_config_value(path)


if __name__ == "__main__":
    main()
