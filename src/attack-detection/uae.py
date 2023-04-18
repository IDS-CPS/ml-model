import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from datetime import datetime
from argparse import ArgumentParser

window_size = 10
target_size = 10
threshold = 6
time_window_threshold = 6

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503"]

attack_df = attack_df.drop(columns=features_dropped)

scaler = joblib.load("scaler/v2/uae.gz")
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))

mean = np.load("npy/uae/mean.npy")
std = np.load("npy/uae/std.npy")
is_attack_period = False
start_period = ""
end_period = ""

consecutive_counter = 0

model = tf.keras.models.load_model("model/v2/uae")
attack_df["Prediction"] = "Normal"

for i in range (len(attack_data)//window_size-1):
    start_index = window_size * i
    end_index = start_index + window_size
    input_window = attack_data[start_index:start_index+window_size]
    target_window = attack_data[end_index:end_index+target_size]

    prediction = model.predict(input_window.reshape(1, window_size, -1), verbose=0).reshape((target_window.shape[0], target_window.shape[1]))
    error = np.abs(prediction - target_window)
    z_score_all = np.abs(error-mean)/std
    z_score_max = np.nanmax(z_score_all, axis=1)

    for j in range(len(z_score_max)):
        if z_score_max[j] > threshold:
            consecutive_counter += 1
        else:
            consecutive_counter = 0

        if consecutive_counter > time_window_threshold:
            consecutive_counter = 0
            start_attack = attack_df.index[end_index+j-time_window_threshold]
            end_attack = attack_df.index[end_index+j+1]
            attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"

attack_df.to_csv("dataset/v2/prediction-uae.csv")