import io
import pickle
import boto3
import gspread
import torch
from delta.tables import DeltaTable
from botocore.exceptions import NoCredentialsError

def upsert_in_delta(source, 
                    sink, 
                    key_cols = None):

        
        if not spark.catalog.tableExists(sink):
                (
                source
                .write
                .option("overwriteSchema", "true")
                .saveAsTable(sink)
                )

        else:
                condition = ' and '.join(['original.' + key + ' = updates.' + key for key in key_cols])
                print(condition)

                original_df = DeltaTable.forName(spark, sink)

                (
                original_df
                .alias('original')
                .merge(source.alias('updates'), condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                ).execute()

        if key_cols is None:
                spark.sql(f'''optimize {sink}''')
        else:
                spark.sql(f'''optimize {sink} zorder by ({', '.join(key_cols)})''')

        spark.sql(f'''vacuum {sink}''')
        


def write_to_gsheet(spreadsheet, worksheet, df, data_range = '!A:J', mode = 'overwrite'):

        """
        1. Provide access of the gsheet to the service email id : "susmit-google-sheet@gheets-tutorial.iam.gserviceaccount.com"
        2. Writes the pandas dataframe to the gsheet (spreadsheetId/worksheet)
        3. Modes available : "overwrite"/ "append"
        """

        client = gspread.service_account(filename="gheets_tutorial_b443e60cc9cc.json")

        try:
                sh = client.open(spreadsheet)
        except gspread.exceptions.SpreadsheetNotFound:
                pass
        
        try:
                wks = sh.worksheet(worksheet)
        except gspread.exceptions.WorksheetNotFound:
                wks = sh.add_worksheet(worksheet, rows=10000, cols=30)

        if mode == "overwrite":
                sh.values_clear(worksheet + "!A1:J10000")
        else:
                pass

        values = [df.columns.values.tolist()] + df.values.tolist()
        sh.values_append(worksheet + data_range, {"valueInputOption": "USER_ENTERED"}, {"values": values})







def save_object_s3(object, filename, bucket_name, project_directory):

        pickle_byte_obj = pickle.dumps(object) 
        s3_resource = boto3.resource('s3')
        key = project_directory + filename
        s3_resource.Object(bucket_name = bucket_name, key = key).put(Body = pickle_byte_obj)



def save_pytorch_model_s3(model, model_name, bucket_name, s3_model_path, local_temp = False):
    
        model_metadata = {'model_class' : model.__class__,'model_params' : getattr(model, 'model_params'), 'model_state_dict' : model.state_dict()}
        
        if local_temp:
                s3 = boto3.client('s3')
                local_model_path = f"{model_name}.pth"
                torch.save(model_metadata, local_model_path)

                try:
                        s3.upload_file(local_model_path, bucket_name, s3_model_path)
                        print(f"Model successfully uploaded to s3://{bucket_name}/{s3_model_path}")
                except FileNotFoundError:
                        print("The file was not found")
                except NoCredentialsError:
                        print("Credentials not available")

        else:
                s3 = boto3.client('s3')
                buffer = io.BytesIO()
                torch.save(model_metadata, buffer)
                buffer.seek(0)

                try:
                        s3.upload_fileobj(buffer, bucket_name, s3_model_path)
                        print(f"Model successfully uploaded to s3://{bucket_name}/{s3_model_path}")
                except NoCredentialsError:
                        print("Credentials not available")
                except Exception as e:
                        print(f"Error uploading the model: {e}")
