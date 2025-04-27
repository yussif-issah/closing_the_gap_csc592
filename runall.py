import argparse
import json
import os
import threading
import time
from itertools import cycle
from itertools import zip_longest



class Worker(threading.Thread):
    def __init__(self, sid, gpu, task_provider):
        threading.Thread.__init__(self)
        self.gpu = gpu
        self.tp = task_provider
        self.id = sid

    def run(self):
        while self.tp.more_tasks():
            cmd_template = self.tp.get_task()
            cmd = cmd_template.format(self.id)

            start = time.time()
            print(f"Runner-{self.id}: {cmd}")

            # set CUDA env and run the actual command
            os.system(
                f"set CUDA_DEVICE_ORDER=PCI_BUS_ID&& "
                f"set CUDA_VISIBLE_DEVICES={self.gpu}&& "
                f"{cmd}"
            )

            elapsed = time.time() - start
            print(f"Runner-{self.id} Execution time: {elapsed/3600:.2f} hours")


class TaskManager:
    def __init__(self, schedule, params, args):
        self.args = args
        
        self.lock = threading.Lock()
        tasks = []

        attack = args.attack

        for dataset in schedule["run"][attack]["datasets"]:
            for arch in schedule["run"][attack]["datasets"][dataset]["arch"]:
                arch_flag = False

                for threshold in (schedule["run"][attack]["thresholds"]):
                    oat_flag = False

                    # add modified arch only once
                    if arch_flag:
                        continue

                    for noise_type in schedule["run"][attack]["datasets"][dataset]["noise"].keys():
                        baseline_flag = False
                        if arch_flag or oat_flag:
                            continue

                        for noise_level in reversed(schedule["run"][attack]["datasets"][dataset]["noise"][noise_type]):
                            if oat_flag:
                                continue
                            else:
                                if "_oat" in arch:
                                    oat_flag = True

                            if arch_flag:
                                continue
                            else:
                                if "_pni" in arch or "_rse" in arch or "_at" in arch:
                                    arch_flag = True

                            # add baseline only once
                            if baseline_flag:
                                continue
                            else:
                                if threshold == 0.0 and not oat_flag:
                                    baseline_flag = True

                            exp_folder = f"{args.output_folder}/{attack}/{dataset}/{arch}/{noise_type}/noise_{noise_level}_threshold_{threshold}"

                            if not os.path.exists(f"{exp_folder}") and not args.output and not args.evaluate:
                                os.makedirs(f"{exp_folder}")

                            if os.path.exists(f"{exp_folder}/raw_data.pkl"):
                                if os.path.exists(f"{exp_folder}/output.json"):
                                    if not args.evaluate and not args.override:
                                        continue
                                    elif args.evaluate and args.override:
                                        pass
                                    else:
                                        continue
                                else:
                                    if args.evaluate:
                                        pass
                            else:
                                # add experiment
                                pass

                            epsilon = params["dataset"][dataset]["epsilon"]
                            query_budget = params["dataset"][dataset]["query_budget"]

                            n_images = schedule["run"][attack]["n_images"]

                            pending_task = f"python main.py -a {attack} -n {n_images} -d {dataset} --seed {args.seed} --arch {arch} -t {threshold} -D {noise_type} --noise_level {noise_level} -o {exp_folder} -eps {epsilon} -q {query_budget} -tid {{}}"

                            if args.evaluate:
                                if args.override:
                                    pending_task += " -m evaluate --override"
                                else:
                                    pending_task += " -m evaluate"
                            else:
                                pending_task += " -m attack"

                            if not args.calibration:
                                pending_task += " --no_calibration"

                            tasks.append(pending_task)

        self.tasks = tasks
        self.idx = 0

    def more_tasks(self):
        return self.idx < len(self.tasks)

    def get_task(self):
        # lock
        with self.lock:
            if self.idx >= len(self.tasks):
                return "echo Tasks finished for thread on GPU {}"
            t = self.tasks[self.idx]
            self.idx += 1
            print(f"Dispatched task {self.idx} / {len(self.tasks)}")
            return t


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", "-g", default="0", help="Comma separated list of GPUs to use. Sorted by PCI_BUS_ID.")
    parser.add_argument("--attack", "-a", default=None, choices=["popskipjump", "surfree"], help="Attack to evaluate.")
    parser.add_argument("--attacks_per_gpu", "-p", type=int, default=1, help="Number of attacks to be run on a single GPU.")
    parser.add_argument("--seed", type=int, default=2023, help="Seed to randomize each attack.")

    parser.add_argument("--config_path", default="configs")
    parser.add_argument("--output_folder", default="output", help="Specifies the output path.")

    parser.add_argument('--output', default=False, action=argparse.BooleanOptionalAction, help="Just outputs all command lines to be executed without actually starting them.")
    parser.add_argument('--evaluate', default=False, action=argparse.BooleanOptionalAction, help="Only performs an evaluation of experimental results.")
    parser.add_argument('--override', default=False, action=argparse.BooleanOptionalAction, help="Overrides existing experiments/results.")

    parser.add_argument('--calibration', default=True, action=argparse.BooleanOptionalAction, help="Applies calibration to the output of the model.")

    return parser.parse_args()


def merge(a, b):
    return [x for y in zip_longest(a, b, fillvalue=object) for x in y if x is not object]


if __name__ == "__main__":
    args = get_args()

    assert args.attacks_per_gpu > 0
    assert args.gpus is not None

    schedule = json.load(open(args.config_path + "/runall.json", "r"))
    params = json.load(open(args.config_path + "/main.json", "r"))

    tm = TaskManager(schedule, params, args)

    gpus = [int(split) for split in args.gpus.split(",")]
    tasks_per_gpu = args.attacks_per_gpu

    gpu_cycle = cycle(gpus)
    workers = [Worker(i, next(gpu_cycle), tm) for i in range(tasks_per_gpu * len(gpus))]

    if not args.output:
        for w in workers:
            w.start()
        for w in workers:
            w.join()
