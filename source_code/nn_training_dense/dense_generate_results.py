import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load

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


def get_input_target(data, num_labels, time_window, scaler):
    """
    Generate the training dataset for a given time window
    Args:
        data: The original dataset containing the training features and labels
        num_labels: Number of labels
        time_window: The time window for generating the training dataset
        scaler_save_path: The path for storing the scaler function

    Returns:
        X_out: The training dataset input
        y_out: The training dataset label
        scaler: The scaler function for the input dataset
    """
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION",
                              "ATTACK_PARAMETER"])
    X_out = []
    y_out = []
    df_out = pd.DataFrame()
    attack_ratios = data["ATTACK_RATIO"].unique()
    attack_durations = data["ATTACK_DURATION"].unique()
    for attack_ratio in attack_ratios:
        for attack_duration in attack_durations:
            temp = data.loc[(data["ATTACK_RATIO"] == attack_ratio) &
                            (data["ATTACK_DURATION"] == attack_duration)]
            if temp.shape[0] == 0:
                continue
            temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
            df_out = df_out.append(temp.iloc[time_window-1:, :])
            temp = temp[["ACTIVE_now", "PACKET_now", "ATTACKED"]]
            X = temp.iloc[:, 0:-num_labels]
            y = temp.iloc[:, -num_labels:]
            X = np.asarray(X).astype(np.float)
            y = np.asarray(y).astype(np.float)
            X = scaler.transform(X)

            for i in range(X.shape[0] - time_window + 1):
                X_out.append(X[i:i + time_window])
                y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    X_out = X_out.reshape((X_out.shape[0], X_out.shape[1]*X_out.shape[2]))
    return X_out, y_out, df_out


def load_model_weights(model_path_input, node, metric, mode):
    """
    Load and return the neural network model weight
    Args:
        model_path_input: path to the model
        node: the node ID for loading the model
        metric: the metric to use for loading the model
        mode: the mode of the metric to use for loading the model

    Returns:
        model: The neural network model with loaded weights
        scaler: The scaler function for the input dataset
    """
    saved_model_path = model_path_input + str(node) + '/'
    scaler_path = saved_model_path + "scaler.pkl"
    model_path = saved_model_path + "final_model/"
    logs_path = saved_model_path + "logs/logs.csv"
    logs = pd.read_csv(logs_path)
    metrics = list(logs.columns)
    metrics.remove("epoch")

    if mode == "max":
        logs = logs.sort_values(by=[metric], ascending=False).reset_index(drop=True)
    elif mode == "min":
        logs = logs.sort_values(by=[metric]).reset_index(drop=True)

    epoch = str((int)(logs["epoch"][0]) + 1).zfill(4)
    checkpoint_path = saved_model_path + "checkpoints/all/weights-" + epoch

    model = tf.keras.models.load_model(model_path)
    model.load_weights(checkpoint_path)
    model.summary()
    scaler = load(open(scaler_path, 'rb'))

    return model, scaler


def generate_general_report(train_dataset, test_dataset, model_path_input, time_window, k_list, metric, mode, output_path):
    """
    Generate a report on different metrics like accuracy, loss, etc of the trained model
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        model_path_input: Path to the NN model used for training
        time_window: The time window used for generating the training/testing dataset
        k_list: The list of different k values used for training
        metric: The metric used for loading weights to the model
        mode: The ID of the node for loading model's weight
        output_path: Output path for storing the reports

    Returns:
        --
    """
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    data = pd.DataFrame()

    nodes = list(train_dataset["NODE"].unique())
    #nodes=nodes[0:5]
    for k in k_list:
        for index, node in enumerate(nodes):
            print("node_index: ", index)
            model, scaler = load_model_weights(model_path_input, node, metric, mode)
            train_dataset_node = train_dataset.loc[(train_dataset["NODE"] == node) &
                                                   (train_dataset["ATTACK_PARAMETER"] == k)]
            test_dataset_node = test_dataset.loc[(test_dataset["NODE"] == node) &
                                                   (test_dataset["ATTACK_PARAMETER"] == k)]

            num_labels = 1
            X_train, y_train, _ = get_input_target(train_dataset_node, num_labels, time_window, scaler)
            X_test, y_test, _ = get_input_target(test_dataset_node, num_labels, time_window, scaler)

            row = {"k": k, "node": node}
            evaluate_train = model.evaluate(X_train, y_train, verbose=1, return_dict=True, use_multiprocessing=True)
            evaluate_test = model.evaluate(X_test, y_test, verbose=1, return_dict=True, use_multiprocessing=True)
            row.update(evaluate_train)
            for key, value in evaluate_test.items():
                row["val_" + key] = value

            data = data.append(row, ignore_index=True)

    output_path_general_report = output_path + "general_report_" + metric + '_' + mode + ".csv"
    #prepare_output_directory(output_path)
    dir_name = str(os.path.dirname(output_path_general_report))
    os.system("mkdir -p " + dir_name)
    data.to_csv(output_path_general_report, index=False)
    print("general report: ", data)

    data = data.groupby(['k']).mean().reset_index()
    output_path_mean_report = output_path + "mean_report_" + metric + '_' + mode + ".csv"
    data.to_csv(output_path_mean_report, index=False)
    print("mean report: ", data)


def generate_attack_prediction_vs_time(model_path_input, train_dataset, test_dataset, time_window, k_list, metric, mode, output_path):
    """
    Generate an attack prediction vs time for the trained models on the training/testing dataset
    Args:
        model_path_input: Path to the NN model used for training
        train_dataset: Training dataset
        test_dataset: Testing dataset
        time_window: The time window used for generating the training/testing dataset
        k_list: The list of different k values used for training
        metric: The metric used for loading weights to the model
        mode: The ID of the node for loading model's weight
        output_path: Output path for storing the results

    Returns:
        --
    """
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    train_result = pd.DataFrame()
    test_result = pd.DataFrame()
    train_dataset = train_dataset.sort_values(by=["TIME"])
    test_dataset = test_dataset.sort_values(by=["TIME"])
    nodes = list(train_dataset["NODE"].unique())
    for k in k_list:
        for index, node in enumerate(nodes):
            print("**********************************************************************")
            print("node_index: ", index)
            model, scaler = load_model_weights(model_path_input, node, metric, mode)

            num_labels = 1
            train_dataset_node = train_dataset.loc[(train_dataset["NODE"] == node) &
                                                   (train_dataset["ATTACK_PARAMETER"] == k)]
            X_train, y_train, df_train = get_input_target(train_dataset_node, num_labels, time_window, scaler)
            y_pred_train = np.rint(model.predict(X_train)).astype(int)
            y_train = np.rint(y_train)

            df_train["TRUE"] = y_train
            df_train["PRED"] = y_pred_train
            df_train["TP"] = df_train["ATTACKED"] & df_train["PRED"]
            df_train["FP"] = df_train["TP"] ^ df_train["PRED"]
            train_result = train_result.append(df_train, ignore_index=True)

            test_dataset_node = test_dataset.loc[(test_dataset["NODE"] == node) &
                                                 (test_dataset["ATTACK_PARAMETER"] == k)]
            X_test, y_test, df_test = get_input_target(test_dataset_node, num_labels, time_window, scaler)
            y_pred_test = np.rint(model.predict(X_test)).astype(int)
            y_test = np.rint(y_test)
            df_test["TRUE"] = y_test
            df_test["PRED"] = y_pred_test
            df_test["TP"] = df_test["ATTACKED"] & df_test["PRED"]
            df_test["FP"] = df_test["TP"] ^ df_test["PRED"]
            test_result = test_result.append(df_test, ignore_index=True)

    #prepare_output_directory(output_path)
    os.system("mkdir -p " + output_path)
    train_result_output_path = output_path + "train_result_" + metric + '_' + mode + ".csv"
    train_result.to_csv(train_result_output_path, index=False)

    test_result_output_path = output_path + "test_result_" + metric + '_' + mode + ".csv"
    test_result.to_csv(test_result_output_path, index=False)


def plot_attack_prediction_vs_time(train_result_path, test_result_path, train_output_path, test_output_path, k_list):
    """
    Plot the attack prediction vs time for the training/testing dataset
    Args:
        train_result_path: Path to the attack prediction vs time results generated by generate_attack_prediction_vs_time function for the training dataset
        test_result_path: Path to the attack prediction vs time results generated by generate_attack_prediction_vs_time function for the testing dataset
        train_output_path: Path to store the attack prediction vs time plot for the training dataset
        test_output_path: Path to store the attack prediction vs time plot for the testing dataset
        k_list: A list of different k values for training the nn model

    Returns:
        --
    """
    train_result = load_dataset(train_result_path)
    attack_ratios = list(train_result["ATTACK_RATIO"].unique())
    attack_durations = list(train_result["ATTACK_DURATION"].unique())
    prepare_output_directory(train_output_path)

    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                plot_data = train_result.loc[(train_result["ATTACK_RATIO"] == attack_ratio) &
                                             (train_result["ATTACK_DURATION"] == attack_duration) &
                                             (train_result["ATTACK_PARAMETER"] == k)]
                if plot_data.shape[0] == 0:
                    continue
                plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
                plot_data = plot_data.sort_values(by=["TIME"])
                plt.clf()
                fig, ax = plt.subplots()
                ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
                ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
                ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                myFmt = mdates.DateFormatter('%H')
                ax.xaxis.set_major_formatter(myFmt)
                ax.legend()
                ax.set_xlabel("Time")
                ax.set_ylabel("Attack")
                ax.set_title("Attack Ratio= " + str(attack_ratio) + " - Duration: " + str(attack_duration) + '\n' +
                             "K= " + str(k))

                output_path = train_output_path + "train_attack_prediction_vs_time_" + str(attack_duration) + \
                                    '_attackRatio_' + str(attack_ratio) + '_duration_' +\
                                    str(attack_duration) + "_k_" + str(k) + '.png'
                fig.savefig(output_path)


    test_result = load_dataset(test_result_path)
    attack_ratios = list(test_result["ATTACK_RATIO"].unique())
    attack_durations = list(test_result["ATTACK_DURATION"].unique())
    prepare_output_directory(test_output_path)

    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                plot_data = test_result.loc[(test_result["ATTACK_RATIO"] == attack_ratio) &
                                             (test_result["ATTACK_DURATION"] == attack_duration) &
                                             (test_result["ATTACK_PARAMETER"] == k)]
                if plot_data.shape[0] == 0:
                    continue
                plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
                plot_data = plot_data.sort_values(by=["TIME"])
                plt.clf()
                fig, ax = plt.subplots()
                ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
                ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
                ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                myFmt = mdates.DateFormatter('%H')
                ax.xaxis.set_major_formatter(myFmt)
                ax.legend()
                ax.set_xlabel("Time")
                ax.set_ylabel("Attack")
                ax.set_title("Attack_Ratio= " + str(attack_ratio) + " - Duration: " + str(attack_duration) + '\n' +
                             "K= " + str(k))

                output_path = test_output_path + "test_attack_prediction_vs_time_" + str(attack_duration) + \
                                     '_attackRatio_' + str(attack_ratio) + '_duration_' + \
                                     str(attack_duration) + "_k_" + str(k) + '.png'
                fig.savefig(output_path)


def plot_metric_vs_attack_parameter(mean_report, output_path):
    """
    Plot different metrics values vs k
    Args:
        mean_report: The mean report dataset generated by generate_general_report function
        output_path: Output path for storing the plots

    Returns:
        --
    """
    plt.clf()
    metrics = mean_report.columns.values

    for metric in metrics:
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        plt.clf()
        plt.plot(mean_report["k"], mean_report[metric], label="Train")
        plt.plot(mean_report["k"], mean_report["val_"+metric], label="Test")
        plt.xlabel("Attack Parameter")
        plt.ylabel(metric)
        plt.title(metric + " vs attack parameter")
        plt.legend()
        plt.savefig(output_path + metric + ".png")


def main_general_report(metric, mode, k_list, time_window):
    """
    The main function for calling the generate_general_report function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights
        k_list: The different k values used for training
        time_window: The time window used for generating the training dataset

    Returns:
        --
    """
    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    model_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/saved_model/"
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/report/"

    generate_general_report(train_dataset, test_dataset, model_path, time_window, k_list, metric, mode, output_path)


def main_generate_attack_prediction_vs_time(metric, mode, k_list, time_window):
    """
    The main function for calling the generate_attack_prediction_vs_time function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights
        k_list: The different k values used for training
        time_window: The time window used for generating the training dataset

    Returns:
        --
    """
    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    model_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/saved_model/"
    output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_prediction_vs_time/data/"

    generate_attack_prediction_vs_time(model_path, train_dataset, test_dataset, time_window, k_list, metric, mode, output_path)


def main_plot_attack_prediction_vs_time(metric, mode, k_list):
    """
    The main function for calling the plot_attack_prediction_vs_time function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights
        k_list: The different k values used for training

    Returns:
        --
    """
    train_output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_prediction_vs_time/plot/train/"
    test_output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_prediction_vs_time/plot/test/"
    prepare_output_directory(train_output_path)
    prepare_output_directory(test_output_path)

    train_result_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_prediction_vs_time/data/train_result_" + metric + '_' + mode + ".csv"
    test_result_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_prediction_vs_time/data/test_result_" + metric + '_' + mode + ".csv"

    plot_attack_prediction_vs_time(train_result_path, test_result_path, train_output_path, test_output_path, k_list)


def main_plot_metric_vs_attack_parameter(metric, mode):
    """
    The main function for calling the metric_vs_attack_parameter function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights

    Returns:
        --
    """
    mean_report_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/report/mean_report_" + metric + '_' + mode + ".csv"
    mean_report = pd.read_csv(mean_report_path)

    output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_dense/current_features_aggregate_all_k/Output/attack_vs_k/"
    prepare_output_directory(output_path)

    plot_metric_vs_attack_parameter(mean_report, output_path)


def main():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    #k_list = np.array([0, 0.3, 1])

    k_list = np.round(k_list, 2)
    time_window = 10

    main_general_report(metric, mode, k_list, time_window)
    main_plot_metric_vs_attack_parameter(metric, mode)

    main_generate_attack_prediction_vs_time(metric, mode, k_list, time_window)
    main_plot_attack_prediction_vs_time(metric, mode, k_list)


if __name__ == "__main__":
    main()