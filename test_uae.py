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

features_considered = ['FIT101', 'MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']

df = df[features_considered]

scaler = joblib.load("scaler/uae.gz")
data = scaler.transform(df)
print(data)

TIME_STEPS = 24
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values)//24):
        start_index = time_steps * i
        output.append(values[start_index : start_index + time_steps])

    return np.stack(output)

data_sequence = create_sequences(data)

def z_score(data, prediction):
    difference = np.absolute(data - prediction)
    mean = np.mean(difference, axis=1, keepdims=True)
    std = np.std(difference, axis=1, keepdims=True)

    z = (np.absolute(difference-mean))/std

    return z

for i in range(len(data_sequence)):
    prediction = model.predict(data_sequence[i].reshape((1, 24, 26)), verbose=0)

    z = z_score(data_sequence[i], prediction)
    max_z = np.amax(z, axis=1)
    above_threshold = max_z[max_z > 5]
    anomaly_len = len(above_threshold)
    if (anomaly_len > 1):
        print(f"anomaly in sequence {i}")

    if (anomaly_len > 3):
        print(f"attack in sequence {i}")