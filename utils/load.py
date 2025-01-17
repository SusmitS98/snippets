import io
import pickle
import boto3
import gspread
import joblib
import torch
import pandas as pd
from delta.tables import DeltaTable
from botocore.exceptions import NoCredentialsError


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



def load_pytorch_model_s3(bucket_name, project_directory, model_name, local_temp=False):

        s3 = boto3.client('s3')
        model = None  

        if local_temp:
                local_model_path = f"{model_name}.pth"
                try:
                        s3.download_file(bucket_name, s3_model_path, local_model_path)
                        print(f"Model successfully downloaded from s3://{bucket_name}/{s3_model_path}")
                        model_metadata = torch.load(local_model_path)
                except FileNotFoundError:
                        print("The file was not found")
                except NoCredentialsError:
                        print("Credentials not available")
                except Exception as e:
                        print(f"Error loading model: {e}")   
        else:
                try:
                        model_buffer = io.BytesIO()
                        s3.download_fileobj(bucket_name, s3_model_path, model_buffer)
                        model_buffer.seek(0)
                        model_metadata = torch.load(model_buffer)
                except NoCredentialsError:
                        print("Credentials not available")
                except Exception as e:
                        print(f"Error loading model: {e}")

        model_class = model_metadata['model_class']
        model_params = model_metadata['model_params']
        model_state_dict = model_metadata['model_state_dict']

        model = model_class(**model_params)
        model.load_state_dict(model_state_dict)
        model.eval()

        return model