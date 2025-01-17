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



def load_pytorch_model_s3(bucket_name, project_directory, local_temp=False):

        s3 = boto3.client('s3')
        model = None  # Initialize the model to None in case of any issues
        
        if local_temp:
                # Download the entire model from S3
                local_model_path = "churn_model.pth"
                try:
                s3.download_file(bucket_name, s3_model_path, local_model_path)
                print(f"Model successfully downloaded from s3://{bucket_name}/{s3_model_path}")
                
                model_metadata = torch.load(local_model_path)
                model_class = model_metadata['model_class']
                model_params = model_metadata['model_params']
                model_state_dict = model_metadata['model_state_dict']
                
                # Instantiate the model with the passed parameters
                model = model_class(**model_params)
                
                # Load the state dictionary
                model.load_state_dict(model_state_dict)
                model.eval()
                
                except FileNotFoundError:
                print("The file was not found")
                except NoCredentialsError:
                print("Credentials not available")
                except Exception as e:
                print(f"Error loading model: {e}")
        
        else:
                try:
                # Download the model as an in-memory file object
                model_buffer = io.BytesIO()
                s3.download_fileobj(bucket_name, s3_model_path, model_buffer)
                
                # Move the buffer's position to the start
                model_buffer.seek(0)
                
                model_metadata = torch.load(model_buffer)
                model_class = model_metadata['model_class']
                model_params = model_metadata['model_params']
                model_state_dict = model_metadata['model_state_dict']

                # Instantiate the model with the passed parameters
                model = model_class(**model_params)
                
                # Load the state dictionary into the model
                model.load_state_dict(model_state_dict)
                model.eval()  # Set the model to evaluation mode
                print("Model loaded directly from S3 into memory")
                
                except NoCredentialsError:
                print("Credentials not available")
                except Exception as e:
                print(f"Error loading model: {e}")
        
        return model