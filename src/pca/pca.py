import pandas as pd
import numpy as np
import joblib

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")
df = df[16000:]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)
df = df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503"]

df = df.drop(columns=features_dropped)

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

print("Features used: ", df.columns)
print(len(df.columns))

scaler = MinMaxScaler()
scaler = scaler.fit(df)
joblib.dump(scaler, "scaler/pca.gz")

train_data = scaler.transform(train_df)
test_data = scaler.transform(test_df)

# Fit PCA to training data
pca = PCA(n_components=13)
pca.fit(test_data)
train_pca = pca.transform(test_data)
train_recon = pca.inverse_transform(train_pca)
error = np.abs(train_recon - test_data)

e_mean = np.mean(error, axis=0)
e_std = np.std(error, axis=0)

# Setup attack data
attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

attack_df = attack_df.drop(columns=features_dropped)
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))


window_size = 1
threshold = 5
anomaly_counter = 0
attack_counter = 0
time_window_threshold = 5

tp = 0
fp = 0
consecutive_counter = 0
for i in range(len(attack_df)//window_size):
    start_index = window_size * i
    window = attack_data[start_index:start_index+window_size]

    window_pca = pca.transform(window)
    window_recon = pca.inverse_transform(window_pca)
    error = np.abs(window - window_recon)
    z_score_all = np.abs(error - e_mean)/e_std
    z_score_max = np.nanmax(z_score_all, axis=1)
    
    attack_label = attack_df.iloc[start_index:start_index+window_size]['Normal/Attack'].values
    print(attack_label)
    print(z_score_max)

    