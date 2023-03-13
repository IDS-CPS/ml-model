import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from argparse import ArgumentParser
from scipy import stats

model = tf.keras.models.load_model('model/autoencoder-kravchik-v2')

df = pd.read_csv("dataset/swat_attack.csv", delimiter=";", decimal=",")
df.columns = [column.strip() for column in df.columns]

attack_df = df.loc[df['Normal/Attack'] == 'Attack']
attack_indexes = attack_df.index
print(f"Attacks are on index {attack_indexes.tolist()}")

features_considered = ['FIT101', 'MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']

df = df[features_considered]

scaler = joblib.load("scaler/uae.gz")
data = scaler.transform(df)

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=24):
    output = []
    for i in range(len(values)//time_steps):
        start_index = time_steps * i
        output.append(values[start_index : start_index + time_steps])

    return np.stack(output)

data_sequence = create_sequences(data, 24)
mean = np.load("uae_mean.npy")
std = np.load("uae_std.npy")

for i in range(len(data_sequence)):
    print(f"Predicting index {i*24} to {i*24 + 24 - 1}")
    checked_index = np.arange(i*24, i*24 + 24, 1).tolist()

    mask = np.in1d(checked_index, attack_indexes)
    if True in mask:
        print("Checking attack")

    prediction = model.predict(data_sequence[i].reshape((1, 24, 26)), verbose=0)

    difference = np.abs(data_sequence[i] - prediction).reshape(24, 26)
    z = (np.abs(difference - mean))/std

    z = np.nan_to_num(z, posinf=0, neginf=0)
    print(np.amax(z, axis=1))