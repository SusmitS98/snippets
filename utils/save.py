
import pickle
import boto3
from delta import DeltaTable


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
