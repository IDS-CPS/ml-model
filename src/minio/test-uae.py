from minio import Minio

minio_url = "localhost:9000"
client = Minio(
    minio_url,
    access_key="admin",
    secret_key="password",
    secure=False
)

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import tempfile
import zipfile
import os

def zipdir(path, ziph):
  # Zipfile hook to zip up model folders
  length = len(path) # Doing this to get rid of parent folders
  for root, dirs, files in os.walk(path):
    folder = root[length:] # We don't need parent folders! Why in the world does zipfile zip the whole tree??
    for file in files:
      ziph.write(os.path.join(root, file), os.path.join(folder, file))

def s3_save_file(object, directory, filename):
    with tempfile.TemporaryDirectory() as tempdir:
        joblib.dump(object, f"{tempdir}/{filename}")
        result = client.fput_object("ids", f"{directory}/{filename}", f"{tempdir}/{filename}")
        print(result.object_name)

def s3_save_keras_model(model, model_name):
  with tempfile.TemporaryDirectory() as tempdir:
    model.save(f"{tempdir}/{model_name}")
    # Zip it up first
    zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
    zipdir(f"{tempdir}/{model_name}", zipf)
    zipf.close()
    result = client.fput_object("ids", f"model/{model_name}.zip", f"{tempdir}/{model_name}.zip")
    print(result.object_name)

def s3_get_keras_model(model_name: str) -> tf.keras.Model:
  with tempfile.TemporaryDirectory() as tempdir:
    # Fetch and save the zip file to the temporary directory
    client.fget_object("ids", f"model/{model_name}.zip", f"{tempdir}/{model_name}.zip")
    # Extract the model zip file within the temporary directory
    with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir}/{model_name}")
    # Load the keras model from the temporary directory
    return tf.keras.models.load_model(f"{tempdir}/{model_name}")

model = tf.keras.models.load_model("model/uae-10")
mean = np.load("npy/mean-10.npy")
std = np.load("npy/std-10.npy")
scaler = joblib.load("scaler/uae-10.gz")

s3_save_file(mean, "npy", "mean.gz")
s3_save_file(std, "npy", "std.gz")
s3_save_file(scaler, "scaler", "uae.gz")
s3_save_keras_model(model, "uae")