import io
import joblib
import boto3
import pandas as pd


def read_from_s3(filepath, type = 'spark'):

        
        if type == 'spark':
                try:
                        data = spark.read.format('delta').load(filepath)
                except:
                        data = spark.read.format('parquet').load(filepath)
        elif type == 'pandas':
                data = pd.read_parquet(filepath)
        else:
                pass
        return data


def load_object_s3(filename, bucket_name, project_directory):

        s3 = boto3.client('s3')
        key = project_directory + filename
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)

        bytes_stream = io.BytesIO(s3_object['Body'].read())
        loaded_object = joblib.load(bytes_stream)
        return loaded_object