import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import sys
import random
import os
import tensorflow_probability as tfp

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
    """Load the dataset and change the type of the "TIME" column to datetime.

    Keyword arguments:
    path -- path to the dataset
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def create_benign_data_for_node(data, dates, begin_date, time_step, node, benign_data, benign_data_rows):
    """Create benign dataset for a given dataset and node, and dates.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    dates -- a list of dates to be used for assigning the occupancy status
    begin_date -- the begin date of assignment
    time_step -- the time steps between dates
    node -- the node to be used for assigning the occupancy status
    output_path -- the path for storing the benign dataset
    """
    benign_data["TIME"] = dates
    benign_data["NODE"] = node
    benign_data["LAT"] = list(data.loc[data["NODE"] == node, "LAT"])[0]
    benign_data["LNG"] = list(data.loc[data["NODE"] == node, "LNG"])[0]
    benign_data["ACTIVE"] = 0
    benign_data = benign_data[["NODE", "LAT", "LNG", "TIME", "ACTIVE"]]
    benign_data = benign_data.sort_values(by=["TIME"])

    data_sid = data.loc[data["NODE"] == node]
    data_sid = data_sid.sort_values(by=["TIME"])
    start_time = begin_date
    for index, row in data_sid.iterrows():
        finish_time = row["TIME"]
        benign_data.loc[(benign_data["TIME"] >= row["TIME"]) &
                        (benign_data["TIME"] < row["TIME"]+timedelta(seconds=time_step)), "ACTIVE"] = row["ACTIVE"]

        benign_data.loc[(benign_data["TIME"] >= start_time) &
                        (benign_data["TIME"] < finish_time), "ACTIVE"] = int(not(row["ACTIVE"]))

        start_time = row["TIME"]+timedelta(seconds=time_step)

    benign_data_rows.append(benign_data)


def create_benign_dataset(data, begin_date, end_date, time_step, num_nodes, benign_packet_cauchy, output_path):
    """Create benign dataset for a given dataset. Benign dataset contains the occupancy status of each node starting
    from the begin_date to end_date with the step of time_step. num_nodes will be used to randomly select num_nodes
    nodes from all of the nodes in the original dataset.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    begin_date -- the begin date of assignment
    emd_date -- the end date of assignment
    time_step -- the time steps between dates
    num_nodes -- number of nodes to be selected out the whole nodes in the dataset
    output_path -- the path for storing the benign dataset
    """
    dates = []
    date = begin_date
    while date < end_date:
        dates.append(date)
        date += timedelta(seconds=time_step)

    nodes = list(data["NODE"].unique())
    num_nodes = min(num_nodes, len(nodes))
    nodes = list(random.sample(nodes, k=num_nodes))
    benign_data = pd.DataFrame(columns=["NODE", "LAT", "LNG", "TIME", "ACTIVE", "ATTACKED"])

    manager = Manager()
    benign_data_rows = manager.list([benign_data])


    p = Pool()
    p.starmap(create_benign_data_for_node, product([data], [dates], [begin_date], [time_step],
                                                       nodes, [benign_data], [benign_data_rows]))
    p.close()
    p.join()


    benign_data = pd.concat(benign_data_rows, ignore_index=True)
    benign_data = benign_data.sort_values(by=["NODE", "TIME"])

    packet_dist = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=benign_packet_cauchy[0],
                                                                     scale=benign_packet_cauchy[1],
                                                                     low=0,
                                                                     high=benign_packet_cauchy[2])
    packet = packet_dist.sample([benign_data.shape[0]])
    packet = np.ceil(packet)

    benign_data["PACKET"] = packet
    benign_data["ATTACKED"] = 0
    benign_data.loc[benign_data["ACTIVE"] == 0, "PACKET"] = 0
    benign_data = benign_data[["NODE", "LAT", "LNG", "TIME", "ACTIVE", "PACKET", "ATTACKED"]]
    benign_data.to_csv(output_path, index=False)


def main_generate_benign_data():
    seed = 6
    random.seed(seed)
    original_data = load_dataset(CONFIG.ORIGINAL_DATASET)
    benign_packet_cauchy_df = pd.read_csv(CONFIG.BENIGN_PACKET_DIST)
    benign_packet_cauchy = [benign_packet_cauchy_df["LOCATION"][0], benign_packet_cauchy_df["SCALE"][0],
                            benign_packet_cauchy_df["MAX_PACKET"][0]]


    begin_date = original_data["TIME"][0]
    begin_date = datetime(begin_date.year, begin_date.month, begin_date.day+1, 0, 0, 0)
    end_date = original_data["TIME"][original_data.shape[0]-1]

    original_data = original_data.loc[(original_data["TIME"] >= begin_date) &
                                      (original_data["TIME"] <= end_date)].reset_index(drop=True)

    nodes = list(original_data["NODE"].unique())
    print("len(nodes): ", len(nodes))

    #num_nodes = len(nodes)
    num_nodes = 20
    time_step = 30

    scale_factor = time_step/10
    benign_packet_cauchy = [scale_factor * i for i in benign_packet_cauchy]


    benign_data_output_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/benign_data/benign_data_" +\
                  str(begin_date) + '_' + str(end_date) + "_time_step_" +\
                  str(time_step) + "_num_ids_" + str(num_nodes) + ".csv"
    prepare_output_directory(benign_data_output_path)
    create_benign_dataset(original_data, begin_date, end_date, time_step, num_nodes, benign_packet_cauchy,
                          benign_data_output_path)


if __name__ == "__main__":
    main_generate_benign_data()