import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def extract_acc_racc(args, configuration):
    # create empty dictionaries for experimental results
    arch_dict = lambda: defaultdict(list)
    defense_dict = lambda: defaultdict(arch_dict)
    dataset_dict = lambda: defaultdict(defense_dict)
    attack_dict = lambda: defaultdict(dataset_dict)
    racc = defaultdict(attack_dict)

    arch_dict = lambda: defaultdict(list)
    dataset_dict = lambda: defaultdict(arch_dict)
    defense_dict = lambda: defaultdict(dataset_dict)
    acc = defaultdict(defense_dict)

    arch_dict = lambda: defaultdict(list)
    defense_dict = lambda: defaultdict(arch_dict)
    dataset_dict = lambda: defaultdict(defense_dict)
    attack_dict = lambda: defaultdict(dataset_dict)
    query_dist = defaultdict(attack_dict)

    for attack in configuration["run"]:
        for dataset in configuration["run"][attack]["datasets"]:
            for arch in configuration["run"][attack]["datasets"][dataset]["arch"]:
                for noise_type in configuration["run"][attack]["datasets"][dataset]["noise"].keys():

                    accs = []
                    raccs = []

                    baseline = (np.nan, np.nan)
                    for noise_level in configuration["run"][attack]["datasets"][dataset]["noise"][noise_type]:
                        for threshold in configuration["run"][attack]["thresholds"]:
                            exp_folder = f"{args.output_folder}/{attack}/{dataset}/{arch}/{noise_type}/noise_{noise_level}_threshold_{threshold}"

                            if os.path.exists(f"{exp_folder}/output.json"):
                                out = json.load(open(f"{exp_folder}/output.json", "r"))

                                raccs.append(1 - float(out["filtered_asr"]))
                                accs.append(float(out["clean_acc"]))

                                # for baseline just add noise_level-times the same value
                                if threshold == 0.0 and noise_level == \
                                        configuration["run"][attack]["datasets"][dataset]["noise"][noise_type][-1]:
                                    baseline = (raccs[-1], accs[-1])
                            else:
                                raccs.append(np.nan)
                                accs.append(np.nan)

                    accs = np.array(accs).reshape((int(len(accs) / len(configuration["run"][attack]["thresholds"])),
                                                   len(configuration["run"][attack]["thresholds"])))
                    raccs = np.array(raccs).reshape((int(len(raccs) / len(configuration["run"][attack]["thresholds"])),
                                                     len(configuration["run"][attack]["thresholds"])))

                    accs[:, 0] = baseline[1]
                    raccs[:, 0] = baseline[0]

                    acc[noise_type][dataset][arch] = accs
                    racc[attack][dataset][noise_type][arch] = raccs

    return {"acc": acc, "racc": racc}


def add_no_noise_baseline(inp):
    return np.vstack(([inp[0, 0] for _ in range(inp.shape[1])], inp))


def plot_pareto(attack, dataset, arch, defense, clean_accuracy, robust_accuracy, noise=None, tau=None):
    plt.rcParams.update({'font.size': 28,
                         'axes.labelsize': 28,
                         'legend.fontsize': 28,
                         })
    fig = plt.figure()

    # add threshold of zero as noise level of zero to get a complete baseline
    clean_accuracy = add_no_noise_baseline(clean_accuracy)
    robust_accuracy = add_no_noise_baseline(robust_accuracy)

    noise_accuracy = clean_accuracy[:, -1]
    noise_robust_accuracy = robust_accuracy[:, -1]

    ordered_points = [(acc, racc) for acc, racc in zip(noise_accuracy, noise_robust_accuracy)]
    ordered_points = [i for i in ordered_points if i[1] is not np.nan]
    ordered_points.sort()

    baseline_x = [x for (x, y) in ordered_points]
    baseline_y = [y for (x, y) in ordered_points]

    # plot baseline
    plt.plot(baseline_x, baseline_y, marker=".", markersize=15, linewidth="4", color="black")

    combined = np.dstack((clean_accuracy, robust_accuracy))
    points = []

    ablate = False
    # get indices for chosen tau or nu
    if tau is None and noise is not None:
        combined = combined[noise + 1, :]
        ablate = True
    elif tau is not None and noise is None:
        combined = combined[:, tau]
        ablate = True

    if ablate:
        to_plot = [[], []]

        # add only point for specifc noise / tau level here!
        for y in combined:
            if y[1] is not None:
                to_plot[0].append(y[0])
                to_plot[1].append(y[1])
                points.append((y[0], y[1]))
    else:
        # no ablation
        stacked = combined.reshape(clean_accuracy.size, 2)

        for y in stacked:
            superior = False
            if y[1] is not None:
                if y[0] > baseline_x[-1] or y[1] > np.interp(y[0], baseline_x, baseline_y):
                    superior = True
                    for point in ordered_points:
                        if point[0] > y[0] and point[1] > y[1]:
                            superior = False

                    if superior:
                        points.append((y[0], y[1]))

    points = [i for i in points if i[1] is not np.nan]

    points.sort()
    sorted_x2 = [x for (x, y) in points]
    sorted_y2 = [y for (x, y) in points]

    if not ablate:
        # find only the maximum points to clean up data
        for _ in range(3):
            remove_idc = []
            for ldx, el in enumerate(sorted_x2):
                if ldx + 1 <= len(sorted_x2) - 1 and sorted_x2[ldx] == sorted_x2[ldx + 1]:
                    a = sorted_y2[ldx]
                    b = sorted_y2[ldx + 1]
                    if a >= b:
                        remove_idc.append(ldx + 1)
                    else:
                        remove_idc.append(ldx)

            sorted_x2 = np.delete(np.array(sorted_x2), remove_idc)
            sorted_y2 = np.delete(np.array(sorted_y2), remove_idc)

        # prettify the plots a bit, i.e., close open gaps or connect thresholded points to baseline (as this is also considered as a threshold)
        if dataset == "cifar10":
            if defense == "rnd":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:4])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:4])
            elif defense == "crop":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:2])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:2])
            elif defense == "jpeg":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:3])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:3])
        elif dataset == "cifar100":
            if defense == "rnd":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:4])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:4])
            elif defense == "crop":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:4])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:4])
            elif defense == "jpeg":
                pass
        elif dataset == "imagenet":
            if defense == "rnd":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:4])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:4])

                sorted_x2 = np.insert(sorted_x2, 5, baseline_x[-1])
                sorted_y2 = np.insert(sorted_y2, 5, baseline_y[-1])
            elif defense == "crop":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:6])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:6])
            elif defense == "jpeg":
                sorted_x2 = np.insert(sorted_x2, 0, baseline_x[0:7])
                sorted_y2 = np.insert(sorted_y2, 0, baseline_y[0:7])

    if not ablate:
        width = 4
    else:
        width = 2

    plt.plot(sorted_x2, sorted_y2, color="black", marker=".", linestyle="dashed", linewidth=str(width), markersize=15,
             label=None)

    if not ablate:
        plt.fill(np.append(baseline_x, sorted_x2[::-1]), np.append(baseline_y, sorted_y2[::-1]),
                 color="lightgray")

    plt.grid(True)
    plt.xlabel("CA")
    plt.ylabel("RA")
    plt.tight_layout(pad=0.25)

    if not os.path.exists(f"{args.output_folder}/{attack}/plots"):
        os.makedirs(f"{args.output_folder}/{attack}/plots")

    if ablate:
        plt.savefig(
            f'{args.output_folder}/{attack}/plots/paper_pareto_ablate_{attack}-{dataset}-{arch}-{defense}-{noise}-{tau}.pdf')
    else:
        plt.savefig(
            f'{args.output_folder}/{attack}/plots/paper_pareto_{attack}-{dataset}-{arch}-{defense}.pdf')  # bbox_inches='tight', pad_inches=.25)

    return fig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs")
    parser.add_argument("--output_folder", "-o", default="results")
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    configuration = json.load(open(f"{args.config_path}/runall.json", "r"))
    data = extract_acc_racc(args, configuration)

    datapoints = [[[] for _ in range(3)] for _ in range(3)]

    for type in ["main", "ablation"]:
        for attack in configuration["run"]:
            for dataset in configuration["run"][attack]["datasets"]:
                for arch in configuration["run"][attack]["datasets"][dataset]["arch"]:
                    for noise_type in configuration["run"][attack]["datasets"][dataset]["noise"].keys():

                        acc = np.round(data['acc'][noise_type][dataset][arch], 2)
                        racc = np.round(data['racc'][attack][dataset][noise_type][arch], 2)

                        if "_pni" not in arch and "_rse" not in arch and "_at" not in arch and "_oat" not in arch:
                            ds_idx = 0 if dataset == "cifar10" else 1 if dataset == "cifar100" else 2 if dataset == "imagenet" else -1
                            noise_idx = 0 if noise_type == "rnd" and attack == "popskipjump" else 1 if noise_type == "crop" and attack == "popskipjump" else 2 if noise_type == "jpeg" and attack == "surfree" else -1

                            if ds_idx != -1 or noise_idx != -1:
                                datapoints[noise_idx][ds_idx] = np.vstack(([acc.T], [racc.T])).T

                        if "_pni" not in arch and "_rse" not in arch and not "_at" in arch and not "_oat" in arch:
                            if type == "ablation":
                                fig_ablate1 = None
                                fig_ablate2 = None

                                if dataset == "cifar10":
                                    if noise_type == "rnd":
                                        fig_ablate1 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=2, tau=None)
                                        fig_ablate2 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=None, tau=6)
                                    elif noise_type == "crop":
                                        fig_ablate1 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=4, tau=None)
                                        fig_ablate2 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=None, tau=5)
                                    elif noise_type == "jpeg":
                                        fig_ablate1 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=0, tau=None)
                                        fig_ablate2 = plot_pareto(attack, dataset, arch, noise_type, acc, racc,
                                                                  noise=None, tau=8)
                                    if args.debug:
                                        if fig_ablate1 is not None:
                                            fig_ablate1.show()
                                        if fig_ablate2 is not None:
                                            fig_ablate2.show()
                            elif type == "main":
                                fig = plot_pareto(attack, dataset, arch, noise_type, acc, racc)

                                if args.debug:
                                    print("----------------------------------------------------")
                                    print(f"{attack} {dataset} {arch} {noise_type}")
                                    print(acc)
                                    print()
                                    print(racc)

                                    fig.show()
