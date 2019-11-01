import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

training_stats_dir = "training_output/"
std_out_dir = "std/"
output_dir = "figures/"

training_file = "training_stats_{}_{}_{}.json"


def parse_training_output():
    """
    This is for parsing the output json files during training of PBG.

    In my scripts I named the files in the following format:
    training_stats_<mode>_<size>_<run_number>.json

    """
    # modes = ['all', 'mult']
    runs = [1, 2, 3, 4, 5]

    sizes = {}
    sizes['all'] = [1, 2, 4, 8, 16, 32, 64]
    sizes['mult_2'] = [4, 8, 16, 32, 64]
    sizes['mult_4'] = [8, 16, 32, 64]
    sizes['mult_8'] = [16, 32, 64]
    sizes['uniform'] = [1]

    modes = ['all', 'mult_2', 'mult_4', 'mult_8', 'uniform']

    res_dict = defaultdict(list)
    for mode in modes:
        for size in sizes[mode]:
            key = "{}_{}".format(mode, size)
            if mode == 'uniform':
                curr_runs = [1, 2, 3, 4]
            else:
                curr_runs = runs
            for run in curr_runs:
                run_dict = {}
                with open(training_stats_dir + training_file.format(mode, size, run)) as f:
                    lines = f.readlines()
                    for line in lines:
                        curr_dict = {}
                        line_dict = json.loads(line)
                        curr_dict["loss"] = line_dict["stats"]["metrics"]["loss"]
                        curr_dict["mrr"] = line_dict["eval_stats_after"]["metrics"]["mrr"]
                        run_dict[line_dict["epoch_idx"]] = curr_dict
                res_dict[key].append(run_dict)

    res_dict = average_output(res_dict)
    return res_dict


def parse_out():
    """
    Hardcoded method for parsing the nohup.out stdout from each of the nodes I trained on.
    """
    res_dict = defaultdict(dict)
    with open(std_out_dir + "parsed_node0.out") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith("Size"):
                curr_size = line.split()[1]
                mrrs = []
            else:
                toks = line.split()
                mrr = toks[8]
                mrrs.append(float(mrr))
                count += 1

                if count == 5:
                    curr_dict = {}
                    # mrrs = reject_outliers(np.asarray(mrrs))
                    curr_dict["mrr"] = np.mean(mrrs)
                    curr_dict["mrr_std"] = np.std(mrrs)
                    res_dict['all'][float(curr_size)] = curr_dict
                    count = 0

    with open(std_out_dir + "parsed_node1.out") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith("Size"):
                curr_size = line.split()[1]
                mrrs = []
            else:
                toks = line.split()
                mrr = toks[8]
                mrrs.append(float(mrr))
                count += 1

                if count == 5:
                    curr_dict = {}
                    # mrrs = reject_outliers(np.asarray(mrrs))
                    curr_dict["mrr"] = np.mean(mrrs)
                    curr_dict["mrr_std"] = np.std(mrrs)
                    res_dict['mult_2'][float(curr_size)] = curr_dict
                    count = 0

    with open(std_out_dir + "parsed_node2.out") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith("Size"):
                curr_size = line.split()[1]
                mrrs = []
            else:
                toks = line.split()
                mrr = toks[8]
                mrrs.append(float(mrr))
                count += 1

                if count == 5:
                    curr_dict = {}
                    # mrrs = reject_outliers(np.asarray(mrrs))
                    curr_dict["mrr"] = np.mean(mrrs)
                    curr_dict["mrr_std"] = np.std(mrrs)
                    res_dict['mult_4'][float(curr_size)] = curr_dict
                    count = 0

    with open(std_out_dir + "parsed_node3.out") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith("Size"):
                curr_size = line.split()[1]
                mrrs = []
            else:
                toks = line.split()
                mrr = toks[8]
                mrrs.append(float(mrr))
                count += 1

                if count == 5:
                    curr_dict = {}
                    # mrrs = reject_outliers(np.asarray(mrrs))
                    curr_dict["mrr"] = np.mean(mrrs)
                    curr_dict["mrr_std"] = np.std(mrrs)
                    res_dict['mult_8'][float(curr_size)] = curr_dict
                    count = 0

    return res_dict


def average_output(res_dict):
    """
    Average each run for each test case and get std for both training loss and mrr
    """
    new_dict = {}
    for key, runs in res_dict.items():
        run_dict = {}
        for epoch in runs[0].keys():
            curr_dict = {}
            losses = []
            mrrs = []
            for run in runs:
                losses.append(float(run[epoch]['loss']))
                mrrs.append(float(run[epoch]["mrr"]))
            losses = np.asarray(losses)
            mrrs = np.asarray(mrrs)
            losses = reject_outliers(losses)
            # mrrs = reject_outliers(mrrs)
            curr_dict['loss'] = np.mean(losses)
            curr_dict['loss_std'] = np.std(losses)
            curr_dict['mrr'] = np.mean(mrrs)
            curr_dict['mrr_std'] = np.std(mrrs)
            run_dict[epoch] = curr_dict
        new_dict[key] = run_dict
    return new_dict


def reject_outliers(data, m=1):
    """
    Use this to remove bad runs from your experiments
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def parse_stdout(file_name):
    """
    Filters the nohup.out stdout so only the final evaluation MRR is returned.
    """
    with open(std_out_dir + file_name) as f:
        lines = f.readlines()
        filtered = filter(lambda line: "[Evaluator] Stats: pos_rank:" in line or line.startswith("Size:"), lines)

    with open(std_out_dir + "parsed_" + file_name, 'w') as f:
        f.writelines(filtered)


def plot_res(res_dict, save_fig=False, fig_name=None, show=True):
    """
    Plots your dictionary of results. Can plot training loss, loss_std, mrr, and mrr_std.
    """
    for key, value in res_dict.items():
        # if key != "all_1" and key != "all_2" and key != "all_4" and key != "all_8" and key != "uniform_1":
        #     continue
        # if key != "mult_2_4" and key != "mult_2_8" and key != "mult_2_16" and key != "uniform_1":
        #     continue
        # if key != "mult_8_16" and key != "mult_8_32" and key != "mult_8_64" and key != "uniform_1":
        #     continue
        losses = [v['loss'] for v in value.values()]
        losses_std = [v['loss_std'] for v in value.values()]
        mrrs = [v['mrr'] for v in value.values()]
        mrrs_std = [v['mrr_std'] for v in value.values()]
        # plt.errorbar(value.keys(), mrrs, mrrs_std, linestyle=None, marker='.', label=key, capsize=5)
        plt.plot(value.keys(), mrrs_std, label=key)
        #
        if "mult" in key:
            style = "--"
        elif "uniform" in key:
            style = ":"
        else:
            style = "-"
        # plt.plot(value.keys(), mrrs_std, label=key, linestyle=style)
    plt.xlim([0, 15])
    # plt.ylim([0.01, 100000])
    # plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("MRR")
    plt.legend()
    if save_fig:
        plt.savefig(output_dir + fig_name)
    if show:
        plt.show()


def plot_test(res_dict):
    """
    Plot your final evaluation results in bar graph form
    """
    offset = 0
    width = .2
    curr_offset = - (5 / 4) * width
    not_set = True
    fig, ax = plt.subplots()

    sizes = {}
    sizes['all'] = [1, 2, 3, 4, 5, 6, 7]
    sizes['mult_2'] = [3, 4, 5, 6, 7]
    sizes['mult_4'] = [4, 5, 6, 7]
    sizes['mult_8'] = [5, 6, 7]
    for key, value in res_dict.items():
        print(key)
        print(value)
        mrrs = [v['mrr'] for v in value.values()]
        mrrs_std = [v['mrr_std'] for v in value.values()]
        xs = np.asarray(sizes[key])
        ax.bar(xs + curr_offset, mrrs, width=width, label=key, yerr=mrrs_std)
        curr_offset += width

        if not_set:
            ax.set_xticks(xs)
            ax.set_xticklabels(value.keys())

            not_set = False
    plt.legend()
    plt.show()


def main():
    res_dict = parse_training_output()
    plot_res(res_dict)


if __name__ == "__main__":
    main()
