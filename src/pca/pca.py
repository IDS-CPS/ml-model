import pandas as pd
import numpy as np
import joblib

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

window_size = 3
threshold = 6
time_window_threshold = 6

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
    print("Attack {}".format(i))
    print("Attack Period : {} - {}".format(attack_dict[i]["start_period"], attack_dict[i]["end_period"]))

    attack_duration = count_time(attack_dict[i]["start_period"].strip(), attack_dict[i]["end_period"].strip())
    total_attack_duration += attack_duration.seconds
    print("Attack Duration: {}".format(attack_duration))
    if len(attack_dict[i]["attacks"]) > 0:
        attacks = attack_dict[i]["attacks"]
        attacks_detected += 1
        print("Detection speed: {}".format(count_time(attack_dict[i]["start_period"].strip(), attacks[0]["end"].strip())))
        print("Time period of attack that is detected: {} - {} ({})".format(
            attacks[0]["start"].strip(), 
            attacks[len(attacks)-1]["end"].strip(),
            count_time(attacks[0]["start"].strip(), attacks[len(attacks)-1]["end"].strip())
        ))
        true_positive_count += len(attacks)

        for attack in attacks:
            print("Time Detected: {} - {}".format(attack["start"].strip(), attack["end"].strip()))
            print("Z Scores: {}".format(attack["z_scores"]))
    print()
    print()

false_alarm_count = 0
for i in range(1, normal_number+1):
    print("Normal Period {}: {} - {}".format(i, normal_dict[i]["start_period"], normal_dict[i]["end_period"]))
    if len(normal_dict[i]["false_alarms"]) > 0:
        false_alarms = normal_dict[i]["false_alarms"]
        false_alarm_count += len(false_alarms)

        for false_alarm in false_alarms:
            print("Time Detected: {} - {}".format(false_alarm["start"].strip(), false_alarm["end"].strip()))
    print() 
    print()

total_normal = len(attack_df[attack_df['Normal/Attack'] == 'Normal']) * 5
true_positive = true_positive_count*5*time_window_threshold
false_positive = false_alarm_count*5*time_window_threshold
false_negative = total_attack_duration - true_positive_count*5*time_window_threshold
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(total_attack_duration)
fpr = false_positive/total_normal

print(total_attack_duration)
print("Attacks Detected: {} out of {}".format(attacks_detected, attack_number))
print("Anomaly counter: {}".format(anomaly_counter))
print("True Positive Seconds: {}".format(true_positive))
print("False Positive Seconds: {}".format(false_positive))
print("False Negative Seconds: {}".format(false_negative))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(2 * (precision * recall)/(precision + recall)))
print("False Positive Rate: {}".format(fpr))