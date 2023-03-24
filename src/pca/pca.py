import pandas as pd
import numpy as np

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

features_considered = ['FIT101', 'MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']

train_df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")
train_df = train_df[16000:]
train_df = train_df.drop("Normal/Attack", axis=1)
train_df = train_df.drop("Timestamp", axis=1)
train_df = train_df[::5]

train_df = train_df[features_considered]

scaler = MinMaxScaler()
scaler = scaler.fit(train_df)
scaled_features = scaler.transform(train_df)
scaled_train_df = pd.DataFrame(scaled_features, index=train_df.index, columns=train_df.columns)

# Fit PCA to training data
pca = PCA(n_components=13)
pca.fit(scaled_train_df)

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

# Assuming that you have a time series dataset stored in a numpy array X with shape (num_samples, num_features).
# Let's say we want to create windows of size window_size.
window_size = 8

attack_data = scaler.transform(attack_df[features_considered])

threshold = 1.5
anomaly_counter = 0
attack_counter = 0
time_window_threshold = 5

for i in range(len(attack_df) - window_size):
    window = attack_data[i:i+window_size, :]
    window_pca = pca.transform(window)
    window_recon = pca.inverse_transform(window_pca)
    error = np.abs(window - window_recon)
    error_mean = np.mean(error, axis=0)
    error_std = np.std(error, axis=0)
    z_score_all = np.abs(error - error_mean)/error_std
    z_score_max = np.nanmax(z_score_all, axis=1)
    
    anomalies = np.where(z_score_max > threshold)[0]
    attack_label = attack_df.iloc[i:i+window_size]['Normal/Attack'].values
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
                print("Attack detected in window {} - {} with anomalies in {}".format(attack_df.index[i], attack_df.index[i+window_size], anomalies))
                break

print("Anomalies: {}, Attacks: {}".format(anomaly_counter, attack_counter))