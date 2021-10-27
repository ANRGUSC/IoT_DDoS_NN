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
    return data


def upsample_dataset(X, y, num_labels):
    """
    Generate the training dataset by upsampling the minor group
    Args:
        X: Original training dataset features
        y: Original training dataset labels
        num_labels: Number of label columns

    Returns:
        The upsampled training dataset
    """
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

    return X, y


def get_train_dataset_input_output(data, num_labels, time_window, scaler_save_path):
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
    temp = data[["ACTIVE", "PACKET", "ATTACKED"]]
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
    k_list = data["ATTACK_PARAMETER"].unique()
    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                temp = data.loc[(data["ATTACK_RATIO"] == attack_ratio) &
                                (data["ATTACK_DURATION"] == attack_duration) &
                                (data["ATTACK_PARAMETER"] == k)]
                if temp.shape[0] == 0:
                    continue
                temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
                temp = temp[["ACTIVE", "PACKET", "ATTACKED"]]
                X = temp.iloc[:,0:-num_labels]
                y = temp.iloc[:,-num_labels:]
                X = np.asarray(X).astype(np.float)
                y = np.asarray(y).astype(np.float)
                X = scaler.transform(X)

                for i in range(X.shape[0] - time_window + 1):
                    X_out.append(X[i:i + time_window])
                    y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    X_out = X_out.reshape((X_out.shape[0], X_out.shape[1]*X_out.shape[2]))
    X_out, y_out = upsample_dataset(X_out, y_out, num_labels)

    return X_out, y_out, scaler


def get_test_dataset_input_output(data, num_labels, time_window, scaler):
    """
    Generate the testing dataset for a given time window
    Args:
        data: The original dataset containing the training features and labels
        num_labels: Number of labels
        time_window: The time window for generating the training dataset
        scaler: The scaler function for the input dataset

    Returns:
        X_out: The upsampled testing dataset input
        y_out: The upsampled testing dataset label
    """
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION",
                              "ATTACK_PARAMETER"])

    X_out = []
    y_out = []
    attack_ratios = data["ATTACK_RATIO"].unique()
    attack_durations = data["ATTACK_DURATION"].unique()
    k_list = data["ATTACK_PARAMETER"].unique()
    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_duration in attack_durations:
                temp = data.loc[(data["ATTACK_RATIO"] == attack_ratio) &
                                (data["ATTACK_DURATION"] == attack_duration) &
                                (data["ATTACK_PARAMETER"] == k)]
                if temp.shape[0] == 0:
                    continue
                temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
                temp = temp[["ACTIVE", "PACKET", "ATTACKED"]]
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

    return X_out, y_out


def autoencoder(input_shape):
    """
    Generate the neural network model
    Args:
        input_shape: the input shape of the dataset given to the model

    Returns:
        The neural network model

    """
    #encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(input_shape,), activation='tanh'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(32, activation='tanh'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    tf.keras.layers.BatchNormalization()

    #decoder
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(32, activation='tanh'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(input_shape, activation='tanh'))


    #encoder_model = encoder(input_shape)
    #autoencoder_model = decoder(input_shape, encoder_model)
    model.compile(loss='mse', optimizer='adam',
                  metrics=[tf.keras.metrics.Accuracy()])
    model.summary()
    return model


def classification(input_shape, output_shape):
    """
    Generate the classification model
    Args:
        input_shape: the input shape of the dataset given to the model
        output_shape: the output shape of the model

    Returns:
        The classification model
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(input_shape,), activation='tanh', trainable=False))
    tf.keras.layers.BatchNormalization(trainable=False)
    model.add(tf.keras.layers.Dense(32, activation='tanh', trainable=False))
    tf.keras.layers.BatchNormalization(trainable=False)
    model.add(tf.keras.layers.Dense(16, activation='tanh', trainable=False))
    tf.keras.layers.BatchNormalization(trainable=False)
    model.add(tf.keras.layers.Dense(8, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.Recall(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    model.summary()
    return model


def setup_callbacks(saved_model_path):
    """
        Setup the call backs for training
    Args:
        saved_model_path: Path for storing the callbacks results

    Returns:
        A list containing all callbacks
    """
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
    """
    Plot the logs stored by callbacks
    Args:
        logs_path: path to the stored logs
        output_path: output path for plotting the logs

    Returns:
        --
    """
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
    """
    Main function for calling plot_logs function
    Args:
        k_list: A list of the different k values used in training the model

    Returns:
        ---
    """
    all_saved_models_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_autoencoder/Output/saved_model/*"
    for directory in glob.glob(all_saved_models_path):
        print(directory)
        logs_path = directory + "/logs/logs.csv"
        output_path = directory + "/logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)


def main_train_model(k_list):
    """
    Main function for training the model
    Args:
        k_list: A list of the different k values used in training the model

    Returns:
        ---
    """
    seed = 2
    tf.random.set_seed(seed)
    random.seed(seed)

    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset_all = load_dataset(train_dataset_path)
    test_dataset_all = load_dataset(test_dataset_path)
    initial_model_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_autoencoder/Output/initial_model/"
    initial_model_encoder_path = initial_model_path + "encoder/"
    prepare_output_directory(initial_model_encoder_path)
    initial_model_autoencoder_path = initial_model_path + "autoencoder/"
    prepare_output_directory(initial_model_autoencoder_path)
    initial_model_classification_path = initial_model_path + "classification/"
    prepare_output_directory(initial_model_classification_path)

    num_labels = 1
    time_window = 10
    initial_scaler_save_path = initial_model_classification_path + "scaler.pkl"
    X_train, y_train, scaler = get_train_dataset_input_output(train_dataset_all, num_labels, time_window, initial_scaler_save_path)

    autoencoder_model = autoencoder(X_train.shape[1])
    #encoder_model.save(initial_model_encoder_path)
    autoencoder_model.save(initial_model_autoencoder_path)

    classification_model = classification(X_train.shape[1], y_train.shape[1])
    classification_model.save(initial_model_classification_path)

    model_output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training_autoencoder/Output/saved_model/"
    prepare_output_directory(model_output_path)

    nodes = list(train_dataset_all["NODE"].unique())
    #nodes = [9380]

    for node_index, node in enumerate(nodes):
        scaler_save_path = model_output_path + str(node) + "/scaler.pkl"
        prepare_output_directory(scaler_save_path)

        saved_model_path = model_output_path + str(node) + '/'
        prepare_output_directory(saved_model_path)

        callbacks_list = setup_callbacks(saved_model_path)

        train_dataset = train_dataset_all.loc[(train_dataset_all["NODE"] == node)]
        test_dataset = test_dataset_all.loc[(test_dataset_all["NODE"] == node)]

        num_labels = 1
        X_train, y_train, scaler = get_train_dataset_input_output(train_dataset, num_labels, time_window, scaler_save_path)
        X_test, y_test = get_test_dataset_input_output(test_dataset, num_labels, time_window, scaler)


        train_dataset_benign = train_dataset_all.loc[(train_dataset_all["NODE"] == node) & (train_dataset_all["ATTACK_RATIO"] == 0)]
        test_dataset_benign = test_dataset_all.loc[(test_dataset_all["NODE"] == node) & (test_dataset_all["ATTACK_RATIO"] == 0)]
        X_train_benign, y_train_benign = get_test_dataset_input_output(train_dataset_benign, num_labels, time_window, scaler)
        X_test_benign, y_test_benign = get_test_dataset_input_output(test_dataset_benign, num_labels, time_window, scaler)

        autoencoder_model = tf.keras.models.load_model(initial_model_autoencoder_path)
        epochs = 50
        batch_size = 32

        history = autoencoder_model.fit(X_train_benign, X_train_benign, batch_size=batch_size,
                                        validation_data=(X_test_benign, X_test_benign), epochs=epochs,
                                        verbose=1)

        plt.clf()
        plt.plot(history.history["accuracy"], label= "train")
        plt.plot(history.history["val_accuracy"], label= "test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Autoencoder Accuracy")
        plt.legend()
        plt.savefig(saved_model_path + "autoencoder_accuracy.png")

        plt.clf()
        plt.plot(history.history["loss"], label= "train")
        plt.plot(history.history["val_loss"], label= "test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Autoencoder Loss")
        plt.legend()
        plt.savefig(saved_model_path + "autoencoder_loss.png")

        classification_model = tf.keras.models.load_model(initial_model_classification_path)
        print(classification_model.get_weights()[0][1])
        for l1, l2 in zip(classification_model.layers[0:3], autoencoder_model.layers[0:2]):
            l1.set_weights(l2.get_weights())

        print(autoencoder_model.get_weights()[0][1])
        print(classification_model.get_weights()[0][1])
        #for layer in classification_model.layers[0:5]:
        #    layer.trainable = False

        epochs = 25
        batch_size = 32

        history = classification_model.fit(X_train, y_train, batch_size=batch_size,
                                           validation_data=(X_test, y_test), epochs=epochs,
                                           verbose=1, callbacks=callbacks_list)

        classification_model.save(saved_model_path + "final_model")
        logs_path = saved_model_path + "logs/logs.csv"
        output_path = saved_model_path + "logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)
        #break


if __name__ == "__main__":
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    main_train_model(k_list)
    main_plot_logs(k_list)