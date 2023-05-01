import pandas as pd
import numpy as np
import joblib

from itertools import product
from datetime import datetime
from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)
parser.add_argument("-d", "--dataset", default="dataset/swat-2015-data.csv", type=str)
parser.add_argument("-w", "--window", default=10, type=int)
parser.add_argument("-t", "--threshold", default=7, type=int)
parser.add_argument("-th", "--timethreshold", default=6, type=int)

args = parser.parse_args()

print(args)

df = pd.read_csv(args.dataset, delimiter=";", decimal=",")
df = df[16000:]
df = df[::5]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)

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
pca.fit(train_data)
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
# attack_df = attack_df[:1000]

attack_df = attack_df.drop(columns=features_dropped)
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))

window_size = args.window

threshold = args.threshold
time_threshold = args.timethreshold
is_attack_period = False
attack_number = 0
attack_detected = set()

consecutive_counter = 0
attack_df["Prediction"] = "Normal"

for i in range(len(attack_df)-window_size):
    end_index = i + window_size
    window = attack_data[i:end_index]

    window_pca = pca.transform(window)
    window_recon = pca.inverse_transform(window_pca)
    error = np.abs(window - window_recon)

    z_score_all = np.abs(error[-1] - e_mean)/e_std
    z_score_max = np.nanmax(z_score_all)
    
    attack_label = attack_df.iloc[end_index-1]['Normal/Attack']

    print(attack_label, z_score_max)

    if (not is_attack_period and attack_label == 'Attack'):
        is_attack_period = True
        attack_number += 1

    if (is_attack_period and attack_label == 'Normal'):
        is_attack_period = False

    if z_score_max > threshold:
        consecutive_counter += 1

    else:
        if consecutive_counter > time_threshold:
            start_attack = attack_df.index[end_index-consecutive_counter]
            end_attack = attack_df.index[end_index-1]
            attack_df.loc[start_attack:end_attack+1, "Prediction"] = "Attack"

            if attack_label == 'Attack':
                attack_detected.add(attack_number)

        consecutive_counter = 0

if consecutive_counter > time_threshold:
    start_attack = attack_df.index[len(attack_data)-consecutive_counter-1]
    end_attack = attack_df.index[len(attack_data)-1]
    attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"


attack_df = attack_df[history_size+1:]
real_value = attack_df["Normal/Attack"].to_numpy()
real_value[real_value == "Normal"] = 0
real_value[(real_value == "Attack")] = 1

predicted_value = attack_df["Prediction"].to_numpy()
predicted_value[predicted_value == "Normal"] = 0
predicted_value[predicted_value == "Attack"] = 1

real_value = np.array(real_value, dtype=int)
predicted_value = np.array(predicted_value, dtype=int)

print(f"{len(attack_detected)} out of {attack_number} detected")
print(attack_detected)

print("Precision:", precision_score(real_value, predicted_value))
print("Recall:", recall_score(real_value, predicted_value))
print("F1 Score:", f1_score(real_value, predicted_value))
fpr, tpr, thresholds = roc_curve(real_value, predicted_value)
print("AUC:", auc(fpr, tpr))