import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from argparse import ArgumentParser
from scipy import stats

model = tf.keras.models.load_model('model/uae-kravchik')

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

features_considered = ['FIT101', 'MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']

scaler = joblib.load("scaler/uae-v3.gz")
attack_data = scaler.transform(attack_df[features_considered])

window_size = 40

threshold = 2.5
anomaly_counter = 0
attack_counter = 0
time_window_threshold = 5

for i in range(len(attack_df)//window_size):
    if (i > 0):
        old_index = start_index
    
    start_index = window_size * i
    window = attack_data[start_index:start_index+window_size, :]

    if (i > 0):
        error = np.abs(prediction - window)
        error_mean = np.mean(error, axis=0)
        error_std = np.std(error, axis=0)
        z_score_all = np.abs(error - error_mean)/error_std
        z_score_max = np.nanmax(z_score_all, axis=1)
        
        anomalies = np.where(z_score_max > threshold)[0]
        attack_label = attack_df.iloc[old_index:old_index+window_size]['Normal/Attack'].values
        print(attack_label)
        print(z_score_max)

        if len(anomalies) > 0:
            anomaly_counter += 1
            consecutive_counter = 0
            expected_index = anomalies[0] + 1
            for j in range(1, len(anomalies)):
                if anomalies[j] == expected_index:
                    consecutive_counter += 1
                    expected_index += 1
                else:
                    consecutive_counter = 0
                    expected_index = anomalies[j] + 1

                if consecutive_counter > time_window_threshold:
                    attack_counter += 1
                    print("Attack detected in window {} - {} with anomalies in {}".format(attack_df.index[old_index], attack_df.index[old_index+window_size], anomalies))
                    break

    prediction = model.predict(window.reshape((1, window.shape[0], window.shape[1]))).reshape((window.shape[0], window.shape[1]))

print("Attack counter: {}".format(attack_counter))
print("Anomaly counter: {}".format(anomaly_counter))