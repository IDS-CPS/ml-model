import pandas as pd
import numpy as np
import joblib

from itertools import product
from datetime import datetime
from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def count_time(start, end):
    start_datetime = datetime.strptime(start, "%d/%m/%Y %I:%M:%S %p")
    end_datetime = datetime.strptime(end, "%d/%m/%Y %I:%M:%S %p")
    return end_datetime - start_datetime

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
pca.fit(train_data)
train_pca = pca.transform(test_data)
train_recon = pca.inverse_transform(train_pca)
error = np.abs(train_recon - test_data)

e_mean = np.mean(error, axis=0)
e_std = np.std(error, axis=0)

# print(e_std)

# Setup attack data
attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]
# attack_df = attack_df[:1000]

attack_df = attack_df.drop(columns=features_dropped)
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))

window_params = [1, 3, 5, 7, 10]
threshold_params = [6, 7, 9]
time_threshold_params = [5, 6, 7]

params = product(window_params, threshold_params, time_threshold_params)
total_normal = len(attack_df[attack_df['Normal/Attack'] == 'Normal']) * 5
total_attack = len(attack_df[attack_df['Normal/Attack'] == 'Attack']) * 5
metrics = []
for param in params:
    window_size = param[0]
    threshold = param[1]
    time_window_threshold = param[2]

    is_attack_period = False
    start_period = ""
    end_period = ""

    attack_dict = dict()
    normal_dict = dict()
    attack_number = 0
    anomaly_counter = 0
    normal_number = 1
    consecutive_counter = 0

    normal_dict[normal_number] = {
        "start_period": attack_df.index[0],
        "false_alarms": []
    }

    for i in range(len(attack_df)//window_size):
        start_index = window_size * i
        window = attack_data[start_index:start_index+window_size]

        window_pca = pca.transform(window)
        window_recon = pca.inverse_transform(window_pca)
        error = np.abs(window - window_recon)
        z_score_all = np.abs(error - e_mean)/e_std
        z_score_max = np.nanmax(z_score_all, axis=1)
        
        attack_label = attack_df.iloc[start_index:start_index+window_size]['Normal/Attack'].values

        for j in range(len(z_score_max)):
            if (not is_attack_period and attack_label[j] == 'Attack'):
                is_attack_period = True
                start_period = attack_df.index[start_index+j]
                attack_number += 1
                attack_dict[attack_number] = {
                    "start_period": start_period,
                    "attacks": []
                }
                normal_dict[normal_number]["end_period"] = attack_df.index[start_index+j-1]

            if (is_attack_period and attack_label[j] == 'Normal'):
                end_period = attack_df.index[start_index+j-1]
                attack_dict[attack_number]["end_period"] = end_period
                is_attack_period = False
                normal_number += 1
                normal_dict[normal_number] = {
                    "start_period": attack_df.index[start_index+j],
                    "false_alarms": []
                }

            if z_score_max[j] > threshold:
                consecutive_counter += 1
                anomaly_counter += 1
            else:
                consecutive_counter = 0

            if consecutive_counter > time_window_threshold:
                consecutive_counter = 0
                if 'Attack' in attack_label:
                    attack_dict[attack_number]["attacks"].append({
                        "start": attack_df.index[start_index+j-time_window_threshold],
                        "end": attack_df.index[start_index+j],
                        "z_scores": z_score_max
                    })
                else:
                    normal_dict[normal_number]["false_alarms"].append({
                        "start": attack_df.index[start_index+j-time_window_threshold],
                        "end": attack_df.index[start_index+j],
                        "z_scores": z_score_max
                    })

    if is_attack_period:
        attack_dict[attack_number]["end_period"] = attack_df.index[len(attack_data)-1]

    if not is_attack_period:
        normal_dict[normal_number]["end_period"] = attack_df.index[len(attack_data)-1]

    attacks_detected = 0
    total_attack_duration = 0
    true_positive_count = 0
    for i in range(1, attack_number+1):
        attack_duration = count_time(attack_dict[i]["start_period"].strip(), attack_dict[i]["end_period"].strip())
        total_attack_duration += attack_duration.seconds

        if len(attack_dict[i]["attacks"]) > 0:
            attacks = attack_dict[i]["attacks"]
            attacks_detected += 1
            true_positive_count += len(attacks)

    false_alarm_count = 0
    for i in range(1, normal_number+1):
        if len(normal_dict[i]["false_alarms"]) > 0:
            false_alarms = normal_dict[i]["false_alarms"]
            false_alarm_count += len(false_alarms)

    true_positive = true_positive_count*5*time_window_threshold
    false_positive = false_alarm_count*5*time_window_threshold
    false_negative = total_attack_duration - true_positive_count*5*time_window_threshold
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(total_attack_duration)
    fpr = false_positive/total_normal
    fnr = false_negative/total_attack
    f1_score = 2 * (precision * recall)/(precision + recall)

    metrics.append([param[0], param[1], param[2], attacks_detected, anomaly_counter, precision, recall, f1_score, fnr, fpr])
 
df_metrics = pd.DataFrame(metrics, columns=[
    'Window Size',
    'Threshold',
    'Time Threshold',
    f'Attacks Detected (out of {attack_number})',
    'Anomaly Counter',
    'Precision',
    'Recall',
    'F1 Score',
    'FNR',
    'FPR'
])

df_metrics.sort_values('F1 Score', ascending=False, inplace=True)
print(df_metrics.to_string())