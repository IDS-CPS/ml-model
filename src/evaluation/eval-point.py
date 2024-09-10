import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve

metrics = []
for filename in os.listdir("dataset/prediction-swat"):
    df = pd.read_csv(f"dataset/prediction-swat/{filename}")

    y_true = df["Normal/Attack"].to_numpy()
    y_pred = df["Prediction"].to_numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    metrics.append({
        "file": filename,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": auc(fpr, tpr)
    })

df = pd.DataFrame.from_dict(metrics)
df = df.sort_values(by=["f1_score"], ascending=False)
print(df.to_string())