import sys
import tensorflow as tf
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def plot_metric_vs_attack_parameter(all_mean_report_path, mean_report_legends, output_path):
    """Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    """

    mean_report = pd.read_csv(all_mean_report_path[0])
    metrics = mean_report.columns.values
    metrics=["binary_accuracy", "recall"]
    colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "yellow", 5: "magenta"}
    line_styles = {0: "solid", 1: "dashed"}
    markers = {0: '.', 1: 'v', 2: '+', 3: 'x', 4: 's', 5: '^'}


    for metric in metrics:
        plt.clf()
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(mean_report["k"], mean_report[metric], color= colors[i], linestyle=line_styles[0],
                     marker=markers[i], label=mean_report_legends[i])
            #plt.plot(mean_report["k"], mean_report["val_" + metric], color=colors[i], linestyle=line_styles[1],
            #         marker=markers[i], label="Test_" + mean_report_legends[i])
        plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
        plt.ylabel(metric)
        if metric == "binary_accuracy":
            plt.ylabel("Binary Accuracy")
            plt.ylim([0.7, 1.02])
        elif metric == "recall":
            plt.ylabel("Recall")
            plt.ylim([0.23, 1.02])
        #plt.title(metric + " vs attack parameter")
        plt.legend()
        plt.savefig(output_path + metric + "_train.png")


    for metric in metrics:
        plt.clf()
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(mean_report["k"], mean_report["val_"+metric], color= colors[i], linestyle=line_styles[1],
                     marker=markers[i], label=mean_report_legends[i])
        plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
        plt.ylabel(metric)
        if metric == "binary_accuracy":
            plt.ylabel("Binary Accuracy")
            plt.ylim([0.7, 1.02])
        elif metric == "recall":
            plt.ylabel("Recall")
            plt.ylim([0.23, 1.02])
        #plt.title(metric + " vs attack parameter")
        plt.legend()
        plt.savefig(output_path + metric + "_test.png")


def main_plot_compare_current(metric, mode):
    """Main function for plotting the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    metric -- the metric for comparing the NN models like binary accuracy, recall
    mode -- the mode for the selected metric
    """
    mean_report_path = [CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/Output/report/mean_report_" + metric + '_' + mode + ".csv",
                        CONFIG.OUTPUT_DIRECTORY + "nn_training_cnn/Output/report/mean_report_" + metric + '_' + mode + ".csv",
                        CONFIG.OUTPUT_DIRECTORY + "nn_training_lstm/Output/report/mean_report_" + metric + '_' + mode + ".csv",
                        CONFIG.OUTPUT_DIRECTORY + "nn_training_autoencoder/Output/report/mean_report_" + metric + '_' + mode + ".csv"]
    mean_report_legends = {0: 'MLP', 1: 'CNN', 2: 'LSTM', 3: 'AEN'}
    output_path = CONFIG.OUTPUT_DIRECTORY + "nn_model_analysis/Output/compare_model/"
    prepare_output_directory(output_path)
    plot_metric_vs_attack_parameter(mean_report_path, mean_report_legends, output_path)


def main():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"

    main_plot_compare_current(metric, mode)


if __name__ == "__main__":
    main()