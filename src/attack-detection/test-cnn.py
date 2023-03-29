import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('model/1d-cnn')

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503"]

attack_df = attack_df.drop(columns=features_dropped)

scaler = joblib.load("scaler/1d-cnn.gz")
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))

window_size = 40

threshold = 4
anomaly_counter = 0
attack_counter = 0
time_window_threshold = 6

tp = 0
fp = 0
consecutive_counter = 0
for i in range(len(attack_df)//window_size):
    if (i > 0):
        old_index = start_index
    
    start_index = window_size * i
    window = attack_data[start_index:start_index+window_size, :]

    if (i > 0):
        error = np.abs(prediction - window)
        error_mean = np.mean(error, axis=1)
        error_std = np.std(error, axis=1)
        z_score_all = np.abs(error - error_mean.reshape((error_mean.shape[0], 1)))/error_std.reshape((error_std.shape[0], 1))
        z_score_max = np.nanmax(z_score_all, axis=1)

        attack_label = attack_df.iloc[old_index:old_index+window_size]['Normal/Attack'].values
        print(attack_label)
        print(z_score_max)

        for j in range(len(z_score_max)):
            if z_score_max[j] > threshold:
                consecutive_counter += 1
                anomaly_counter += 1
            else:
                consecutive_counter = 0

            if consecutive_counter > time_window_threshold:
                attack_counter += 1
                consecutive_counter = 0
                print("Attack detected in window {} - {} at time {} with anomalies in {}".format(attack_df.index[start_index+j], attack_df.index[start_index+window_size-1], attack_df.index[start_index+j], z_score_max))

                if 'Attack' in attack_label:
                    print("Attack correctly detected")
                    tp += 1
                else:
                    fp += 1

    prediction = model.predict(window.reshape((1, window.shape[0], window.shape[1]))).reshape((window.shape[0], window.shape[1]))

print("Attack counter: {}".format(attack_counter))
print("Anomaly counter: {}".format(anomaly_counter))
print(tp, fp)