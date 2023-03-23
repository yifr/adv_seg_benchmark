import os
import boto3
import json

def download_data(bucket_name):
    s3 = boto3.resource("s3")

    datasets = ["gestalt_shapegen", "tdw", "nsd", "hypersim_v3"]
    passes = ["images", "masks"]

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        path, filename = os.path.split(obj.key)

        if len(path.split("/")) == 1:
            continue

        dataset = path.split("/")[0]
        image_pass = path.split("/")[1]

        if dataset in datasets and image_pass in passes and filename.endswith(".png"):
            print(path, filename)
            if not os.path.exists(path):
                os.makedirs("data/" + path, exist_ok=True)
            bucket.download_file(obj.key, "data/" + obj.key)

download_data("mlve-v1")