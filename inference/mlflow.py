import mlflow

def apply_mlflow_model_udf(model_run_id, result_type = None):

        """
        Usage :  
                1. Make Sure df has all the feature cols.
                2. df = df.withColumn('col_name', apply_mlflow_model_udf(model_run_id))
                3. result_type : types belonging to pyspark.sql.types
        """

        model = mlflow.sklearn.load_model(f"runs:/{model_run_id}/model")
        udf = mlflow.pyfunc.spark_udf(get_spark(), f"runs:/{model_run_id}/model", result_type = result_type)
        feature_name = model.feature_names_in_

        return udf(*feature_name)