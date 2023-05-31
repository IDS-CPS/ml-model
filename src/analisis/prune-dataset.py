import pandas as pd

df = pd.read_csv("dataset/pompa-train-v2.csv")

features_considered = ['adc_level', 'adc_flow', 'adc_pressure_left', 'adc_pressure_right']
df = df[features_considered]

df.to_csv("dataset/pompa-train-v2.csv", index=False)