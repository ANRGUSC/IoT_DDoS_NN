import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime

sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def upsample_dataset(X, y, num_labels):
    shape = X.shape
    X = X.reshape((shape[0], shape[1]*shape[2]))

    df = pd.DataFrame(X)
    df["ATTACKED"] = y
    print(df["ATTACKED"].value_counts())
    attacked_data = df.loc[df["ATTACKED"] == 1]
    not_attacked_data = df.loc[df["ATTACKED"] == 0]
    attacked_data = resample(attacked_data, replace=True, n_samples=not_attacked_data.shape[0], random_state=10)
    df = pd.concat([not_attacked_data, attacked_data])
    print(df["ATTACKED"].value_counts())

    X = np.array(df.iloc[:,0:-num_labels])
    y = np.array(df.iloc[:,-num_labels:])
    X = X.reshape((X.shape[0], shape[1], shape[2]))

    return X, y


def get_train_dataset_input_output(data, num_labels, time_window, scaler_save_path):
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION",
                              "ATTACK_PARAMETER"])
    temp = data[["ACTIVE_now", "PACKET_now", "ATTACKED"]]
    X = temp.iloc[:,0:-num_labels]
    y = temp.iloc[:,-num_labels:]
    X = np.asarray(X).astype(np.float)
    y = np.asarray(y).astype(np.float)

    scaler = StandardScaler()
    scaler.fit_transform(X)
    dump(scaler, open(scaler_save_path, 'wb'))

    X_out = []
    y_out = []
    attack_ratios = data["ATTACK_RATIO"].unique()
    attack_durations = data["ATTACK_DURATION"].unique()
    k_list  = data["ATTACK_PARAMETER"].unique()
    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                temp = data.loc[(data["ATTACK_RATIO"] == attack_ratio) &
                                (data["ATTACK_DURATION"] == attack_duration) &
                                (data["ATTACK_PARAMETER"] == k)]
                if temp.shape[0] == 0:
                    continue
                temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
                temp = temp[["ACTIVE_now", "PACKET_now", "ATTACKED"]]
                X = temp.iloc[:,0:-num_labels]
                y = temp.iloc[:,-num_labels:]
                X = np.asarray(X).astype(np.float)
                y = np.asarray(y).astype(np.float)
                X = scaler.transform(X)

                for i in range(X.shape[0] - time_window + 1):
                    X_out.append(X[i:i + time_window])
                    y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    X_out, y_out = upsample_dataset(X_out, y_out, num_labels)

    return X_out, y_out, scaler


def get_test_dataset_input_output(data, num_labels, time_window, scaler):
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION",
                              "ATTACK_PARAMETER"])

    X_out = []
    y_out = []
    attack_ratios = data["ATTACK_RATIO"].unique()
    attack_durations = data["ATTACK_DURATION"].unique()
    k_list  = data["ATTACK_PARAMETER"].unique()
    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                temp = data.loc[(data["ATTACK_RATIO"] == attack_ratio) &
                                (data["ATTACK_DURATION"] == attack_duration) &
                                (data["ATTACK_PARAMETER"] == k)]
                if temp.shape[0] == 0:
                    continue
                temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
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
    return X_out, y_out


def create_nn_model(input_shape, output_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=5, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.Recall(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    model.summary()
    return model


def setup_callbacks(saved_model_path):
    metrics = ['loss', 'accuracy', 'recall', 'true_positives', 'false_positives', 'val_loss',
               'val_accuracy', 'val_recall', 'val_true_positives', 'val_false_positives']


    #checkpoint_path = saved_model_path + "checkpoints/all/weights-{epoch:04d}-{recall:.2f}-{val_recall:.2f}"
    checkpoint_path = saved_model_path + "checkpoints/all/weights-{epoch:04d}"
    prepare_output_directory(checkpoint_path)
    cp_1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True)

    checkpoint_path = saved_model_path + "checkpoints/recall/weights"
    prepare_output_directory(checkpoint_path)
    cp_2 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='recall',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + "checkpoints/val_recall/weights"
    prepare_output_directory(checkpoint_path)
    cp_3 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_recall',
        mode='max',
        save_best_only=True)


    checkpoint_path = saved_model_path + "checkpoints/accuracy/weights"
    prepare_output_directory(checkpoint_path)
    cp_4 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + "checkpoints/val_accuracy/weights"
    prepare_output_directory(checkpoint_path)
    cp_5 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + "checkpoints/loss/weights"
    prepare_output_directory(checkpoint_path)
    cp_6 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    checkpoint_path = saved_model_path + "checkpoints/val_loss/weights"
    prepare_output_directory(checkpoint_path)
    cp_7 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    log_path = saved_model_path + "logs/logs.csv"
    prepare_output_directory(log_path)
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)

    tensorboard_path = saved_model_path + "tensorboard/" + str(datetime.now())
    prepare_output_directory(tensorboard_path)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir= tensorboard_path, histogram_freq=1)

    callbacks = [cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, csv_logger, tensorboard]
    #callbacks = [cp_1, csv_logger]
    return callbacks


def plot_logs(logs_path, output_path):
    metrics = ['loss', 'accuracy', 'recall', 'true_positives', 'false_positives', 'val_loss',
               'val_accuracy', 'val_recall', 'val_true_positives', 'val_false_positives']
    metrics = ['loss', 'accuracy', 'recall', 'true_positives', 'false_positives']
    logs = pd.read_csv(logs_path)
    metrics = logs.columns.values
    new_metrics = {}
    for metric in metrics:
        if metric[-2] == '_':
            new_metrics[metric] = metric[:-2]
        elif metric[-3] == '_':
            new_metrics[metric] = metric[:-3]

    logs = logs.rename(new_metrics, axis="columns")
    metrics = logs.columns.values

    for metric in metrics:
        if metric == "epoch" or "val" in metric:
            continue
        plt.clf()
        plt.plot(logs["epoch"], logs[metric], label="Train")
        plt.plot(logs["epoch"], logs["val_"+metric], label="Test")
        plt.xlabel("Epoch Number")
        plt.ylabel(metric)
        plt.title(metric + " vs epoch")
        plt.legend()
        plt.savefig(output_path + metric + ".png")


def main_plot_logs(k_list):
    all_saved_models_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_cnn/current_features_aggregate_all_k/Output/saved_model/*"
    for directory in glob.glob(all_saved_models_path):
        print(directory)
        logs_path = directory + "/logs/logs.csv"
        output_path = directory + "/logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)


def main_train_model(k_list):
    seed = 2
    tf.random.set_seed(seed)
    random.seed(seed)

    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset_all = load_dataset(train_dataset_path)
    test_dataset_all = load_dataset(test_dataset_path)
    initial_model_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_cnn/current_features_aggregate_all_k/Output/initial_model/"
    prepare_output_directory(initial_model_path)

    num_labels = 1
    time_window = 10
    initial_scaler_save_path = initial_model_path + "scaler.pkl"
    X_train, y_train, scaler = get_train_dataset_input_output(train_dataset_all, num_labels, time_window, initial_scaler_save_path)
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    model = create_nn_model(input_shape, output_shape)
    model.save(initial_model_path)

    model_output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_cnn/current_features_aggregate_all_k/Output/saved_model/"
    prepare_output_directory(model_output_path)

    nodes = list(train_dataset_all["NODE"].unique())
    #nodes=nodes[0:5]

    for node_index, node in enumerate(nodes):
        scaler_save_path = model_output_path + str(node) + "/scaler.pkl"
        prepare_output_directory(scaler_save_path)

        saved_model_path = model_output_path + str(node) + '/'
        prepare_output_directory(saved_model_path)

        callbacks_list = setup_callbacks(saved_model_path)

        train_dataset = train_dataset_all.loc[(train_dataset_all["NODE"] == node)]
        test_dataset = test_dataset_all.loc[(test_dataset_all["NODE"] == node)]

        train_dataset = train_dataset.sort_values(by=["TIME"]).reset_index(drop=True)
        test_dataset = test_dataset.sort_values(by=["TIME"]).reset_index(drop=True)

        X_train, y_train, scaler = get_train_dataset_input_output(train_dataset, num_labels, time_window, scaler_save_path)
        X_test, y_test = get_test_dataset_input_output(test_dataset, num_labels, time_window, scaler)
        print("train: ", X_train.shape, y_train.shape)
        print("test: ", X_test.shape, y_test.shape)

        model = tf.keras.models.load_model(initial_model_path)

        epochs = 25
        batch_size = 32

        history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=epochs,
                            verbose=1, callbacks=callbacks_list)

        model.save(saved_model_path + "final_model")
        logs_path = saved_model_path + "logs/logs.csv"
        output_path = saved_model_path + "logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)


if __name__ == "__main__":
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    #k_list = np.array([0, 0.3, 1])
    main_train_model(k_list)
    main_plot_logs(k_list)