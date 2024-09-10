import pandas as pd
import numpy as np
import os

def precision(tp, fp):
    return tp/(tp+fp)

def recall(tp, fn):
    return tp/(tp+fn)

def f1_score(precision, recall):
    return (2*precision*recall)/(precision+recall)

metrics = []
for filename in os.listdir("dataset/prediction-swat"):
    df = pd.read_csv(f"dataset/prediction-swat/{filename}")
    
    tp = 0
    fp = 0
    fn = 0
    is_attack_period = False
    start_period = 0
    skip_counter = 0
    for index, row in df.iterrows():
        # Check false positives
        if not is_attack_period and row["Normal/Attack"] == 1:
            prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
            data_class = df.loc[start_period:index-1, "Normal/Attack"].to_numpy()
            intersect = np.intersect1d(prediction, data_class)
            # print(prediction, data_class)
            if 1 in prediction:
                fp += 1

            start_period = index
            is_attack_period = True

        if is_attack_period and row["Normal/Attack"] == 0:
            # if skip_counter < 30:
            #     skip_counter += 1
            #     continue

            skip_counter = 0
            prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
            data_class = df.loc[start_period:index-1, "Normal/Attack"].to_numpy()
            intersect = np.intersect1d(prediction, data_class)
            if 1 not in intersect:
                fn += 1
            else:
                tp += 1
            
            is_attack_period = False
            start_period = index

    if is_attack_period:
        prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
        data_class = df.loc[start_period:index-1, "Normal/Attack"].to_numpy()
        intersect = np.intersect1d(prediction, data_class)

        if 1 not in intersect:
            # print(prediction)
            # print(data_class)
            # print(f"fn at {start_real_attack}-{index-1}")
            fn += 1
        else:
            tp += 1

    if not is_attack_period:
        prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
        data_class = df.loc[start_period:index-1, "Normal/Attack"].to_numpy()
        intersect = np.intersect1d(prediction, data_class)
        if 1 in intersect:
            # print(f"fn at {start_real_attack}-{index-1}")
            fp += 1

    precision_metric = precision(tp, fp)
    recall_metric = recall(tp, fn)
    metrics.append({
        "file": filename,
        "precision": precision_metric,
        "recall": recall_metric,
        "f1_score": f1_score(precision_metric, recall_metric),
        "tp": tp,
        "fn": fn,
        "fp": fp
    })
    # break

df = pd.DataFrame.from_dict(metrics)
df = df.sort_values(by=["recall", "f1_score"], ascending=False)
print(df.to_string())

df.to_csv("evaluation-swat.csv")