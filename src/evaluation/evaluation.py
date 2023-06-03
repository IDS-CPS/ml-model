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
for filename in os.listdir("dataset/prediction"):
    df = pd.read_csv(f"dataset/prediction/{filename}")
    
    tp = 0
    fp = 0
    fn = 0
    is_attack_period = False
    real_attack = False
    start_real_attack = 0
    start_period = 0
    skip_counter = 0
    for index, row in df.iterrows():
        # Check false negatives
        if not real_attack and row["Class"] == 1:
            start_real_attack = index
            real_attack = True

        if real_attack and row["Class"] == 0:
            real_attack = False
            prediction = df.loc[start_real_attack:index-1, "Prediction"].to_numpy()
            data_class = df.loc[start_real_attack:index-1, "Class"].to_numpy()
            intersect = np.intersect1d(prediction, data_class)
            if 1 not in intersect:
                # print(prediction)
                # print(data_class)
                # print(f"fn at {start_real_attack}-{index-1}")
                fn += 1

        if not is_attack_period and row["Prediction"] == 1:
            is_attack_period = True
            start_period = index

        if is_attack_period and row["Prediction"] == 1:
            skip_counter = 0

        # Check true positives and false positives
        if is_attack_period and row["Prediction"] == 0:
            if skip_counter > 30:
                is_attack_period = False
                prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
                data_class = df.loc[start_period:index-1, "Class"].to_numpy()
                intersect = np.intersect1d(prediction, data_class)

                if 1 in intersect:
                    print(f"tp at {start_period}-{index-1}")
                    tp += 1
                else:
                    print(f"fp at {start_period}-{index-1}")
                    fp += 1

            skip_counter += 1

    if is_attack_period:
        prediction = df.loc[start_period:index-1, "Prediction"].to_numpy()
        data_class = df.loc[start_period:index-1, "Class"].to_numpy()
        intersect = np.intersect1d(prediction, data_class)

        if 1 in intersect:
            # print(f"tp at {start_period}-{index-1}")
            tp += 1
        else:
            # print(f"fp at {start_period}-{index-1}")
            fp += 1

    if real_attack:
        prediction = df.loc[start_real_attack:index-1, "Prediction"].to_numpy()
        data_class = df.loc[start_real_attack:index-1, "Class"].to_numpy()
        intersect = np.intersect1d(prediction, data_class)
        if 1 not in intersect:
            # print(f"fn at {start_real_attack}-{index-1}")
            fn += 1

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
    break

df = pd.DataFrame.from_dict(metrics)
df = df.sort_values(by=["f1_score", "precision"], ascending=False)
print(df.to_string())