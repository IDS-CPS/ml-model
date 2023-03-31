import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('model/v2/uae')

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

window_size = 10
target_size = 10

threshold = 50
anomaly_counter = 0
time_window_threshold = 6

mean = np.load("npy/uae/mean.npy")
std = np.load("npy/uae/std.npy")
is_attack_period = False
attack_number = 0
start_period = ""
end_period = ""
attacks = []
attack_dict = dict()
for i in range (len(attack_data)//window_size-1):
    start_index = window_size * i
    end_index = start_index + window_size
    input_window = attack_data[start_index:start_index+window_size]
    target_window = attack_data[end_index:end_index+target_size]

    prediction = model.predict(input_window.reshape(1, window_size, -1)).reshape((target_window.shape[0], target_window.shape[1]))
    error = np.abs(prediction - target_window)
    z_score_all = np.abs(error-mean)/std
    z_score_max = np.nanmax(z_score_all, axis=1)

    attack_label = attack_df.iloc[end_index:end_index+target_size]['Normal/Attack'].values
    print(attack_label)
    print(z_score_max)

    for j in range(len(z_score_max)):
        print(attack_label[j])
        if (not is_attack_period and attack_label[j] == 'Attack'):
            is_attack_period = True
            start_period = attack_df.index[end_index+j]
            attack_number += 1
            attack_dict[attack_number] = []

        if (is_attack_period and attack_label[j] == 'Normal'):
            end_period = attack_df.index[end_index+j-1]
            attacks.append(f"{start_period} - {end_period}")
            is_attack_period = False

        if z_score_max[j] > threshold:
            consecutive_counter += 1
            anomaly_counter += 1
        else:
            consecutive_counter = 0

        if consecutive_counter > time_window_threshold:
            consecutive_counter = 0
            attack_dict[attack_number].append(f"{attack_df.index[end_index]} - {attack_df.index[end_index+target_size-1]}")
            print("Attack detected in window {} - {} at time {} with anomalies z-scores {}".format(attack_df.index[end_index], attack_df.index[end_index+target_size-1], attack_df.index[end_index+j], z_score_max))

print("Total attacks: {}".format(attack_number))
print("Attack Periods: {}".format(attacks))

attacks_detected = 0
print(attack_dict)

for i in range(1, attack_number+1):
    if len(attack_dict[i] > 0):
        print("Attack {}: {}".format(i, attack_dict[i]))
        attacks_detected += 1

# for key, value in attack_dict:
#     if len(value) > 0:
#         print("Attack {}: {}".format(key, value))
#         attacks_detected += 1

print("Attacks Detected: {} out of {}".format(attacks_detected, attack_number))
print("Anomaly counter: {}".format(anomaly_counter))