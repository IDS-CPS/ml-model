from minio import Minio
import os
import tensorflow as tf
import requests
import numpy as np
import joblib
import tempfile
import zipfile

minio_url = "10.8.1.146:9000"
client = Minio(
    minio_url,
    access_key="qeB0URQA8q2MkmoRM1zB",
    secret_key="beO0B4CKkMl20bqZYA15yeP093UeLURfbtFIxfWR",
    secure=False
)

def load_joblib(url, r):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(r.content)
        tmp.seek(0)
        test = joblib.load(tmp)
        print(test)

def load_model(r):
    with tempfile.TemporaryDirectory() as td:
        f_name = os.path.join(td, 'tmp.zip')
        open(f_name, 'wb').write(r.content)
        # Extract the model zip file within the temporary directory
        with zipfile.ZipFile(f_name) as zip_ref:
            zip_ref.extractall(f"{td}/tmp-model")
        # Load the keras model from the temporary directory
        return tf.keras.models.load_model(f"{td}/tmp-model")

url = client.presigned_get_object("learning-model-bucket", "scaler/cnn-40.gz")
print(url)
url = client.presigned_get_object("learning-model-bucket", "npy/cnn/mean-40.gz")
print(url)
url = client.presigned_get_object("learning-model-bucket", "npy/cnn/std-40.gz")
print(url)
url = client.presigned_get_object("learning-model-bucket", "model/cnn-40.gz")
print(url)

# r = requests.get(url)

# model = load_model(r)
# print(model.summary())
# test = joblib.load("test.npy")
# print(test)