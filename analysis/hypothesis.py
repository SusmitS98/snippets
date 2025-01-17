import pyspark.sql.types as T
import pyspark.sql.functions as F
from scipy.stats import t, ttest_ind, chi2_contingency





def two_sample_students_t_test(data, treatment_colname, control_value, treatment_value, response_colname, alpha=0.05):

        """
        Perform a two-sample t-test on two sets of data.
        """
        
        
        statistics_dict = (data
                            .select(treatment_colname, response_colname)
                            .agg(
                              F.mean(F.when(F.col(treatment_colname) == control_value, F.col(response_colname)).otherwise(None)).alias('mean_group_cg'), 
                              F.mean(F.when(F.col(treatment_colname) == treatment_value, F.col(response_colname)).otherwise(None)).alias('mean_group_tg'), 
                              F.stddev(F.when(F.col(treatment_colname) == control_value, F.col(response_colname)).otherwise(None)).alias('stddev_group_cg'), 
                              F.stddev(F.when(F.col(treatment_colname) == treatment_value, 
                              F.col(response_colname)).otherwise(None)).alias('stddev_group_tg'), 
                              F.count(F.when(F.col(treatment_colname) == control_value, F.col(response_colname)).otherwise(None)).alias('count_group_cg'), 
                              F.count(F.when(F.col(treatment_colname) == treatment_value, F.col(response_colname)).otherwise(None)).alias('count_group_tg'))
                            .collect()[0]
                            .asDict())

        # Compute delta and percent_delta
        delta = float(statistics_dict["mean_group_tg"] - statistics_dict["mean_group_cg"]) 
        percent_delta = float(100.0 * (delta/statistics_dict["mean_group_cg"]))
        
        # Compute the pooled standard deviation
        pooled_stddev = (((statistics_dict["stddev_group_cg"]**2 * (statistics_dict["count_group_cg"] - 1)) + (statistics_dict["stddev_group_tg"]**2 * (statistics_dict["count_group_tg"] - 1))) / (statistics_dict["count_group_cg"] + statistics_dict["count_group_tg"] - 2))**0.5
        
        # Compute the t-statistic
        se = pooled_stddev * ((1/statistics_dict["count_group_cg"]) + (1/statistics_dict["count_group_tg"]))**0.5
        t_statistic = float(delta/se)

        # Compute degrees of freedom
        degrees_of_freedom = statistics_dict["count_group_cg"] + statistics_dict["count_group_tg"] - 2

        # Compute the p-value (two-tailed)
        p_value = float((1 - t.cdf(abs(t_statistic), degrees_of_freedom)) * 2)

        # Compare p-value with alpha to determine significance
        if p_value < alpha:
            result = "Reject null hypothesis: There is a significant difference between the means of the two groups."
            is_significant = 1
        else:
            result = "Fail to reject null hypothesis: There is no significant difference between the means of the two groups."
            is_significant = 0
            
        # Compute the p-value (two-tailed)
        error = float(t.ppf(1-alpha/2, degrees_of_freedom)*se)
        ci = [delta - error, delta + error] 
        percent_ci = [100.0 * ((delta-error)/statistics_dict["mean_group_cg"]), 100.0 * ((delta+error)/statistics_dict["mean_group_cg"])]
        

        schema = T.StructType([T.StructField('t_statistic', T.DoubleType(), True), 
                              T.StructField('delta', T.DoubleType(), True), 
                              T.StructField('ci', T.ArrayType(T.FloatType()), True), 
                              T.StructField('percent_delta', T.DoubleType(), True), 
                              T.StructField('percent_ci', T.ArrayType(T.FloatType()), True), 
                              T.StructField('p_value', T.DoubleType(), True), 
                              T.StructField('result', T.StringType(), True),
                              T.StructField('is_significant', T.IntegerType(), True)])

        df_out = spark.createDataFrame(data = [dict(zip(["t_statistic", "delta", "ci" , "percent_delta", "percent_ci", "p_value", "result", "is_significant"],
                                                         [t_statistic, delta, ci, percent_delta, percent_ci, p_value, result, is_significant]))], 
                                       schema = schema)
        return df_out
  




def two_sample_chi_square_test(data, treatment_colname, control_value, treatment_value, response_colname, alpha=0.05):

        """
        Perform a chi-square test of independence on categorical data.
        """


        
        data = (data
                .where(f"{treatment_colname} in {(control_value, treatment_value)}")
                .withColumn("is_treatment", F.col(treatment_colname) == F.lit(treatment_value)))

        # Create contingency table
        contingency_table = data.crosstab(treatment_colname, response_colname)

        # Convert contingency table to a list of lists
        contingency_list = contingency_table.collect()

        # Convert the list of lists to a 2D array
        contingency_array = [[int(row[col]) for col in contingency_table.columns[1:]] for row in contingency_list]

        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

        # Compare p-value with alpha to determine significance
        if p_value < alpha:
            result = "Reject null hypothesis: There is a significant relationship between the two categorical variables."
        else:
            result = "Fail to reject null hypothesis: There is no significant relationship between the two categorical variables."
            
        schema = T.StructType([T.StructField('chi2_statistic', T.DoubleType(), True),
                              T.StructField('p_value', T.DoubleType(), True), 
                              T.StructField('result', T.StringType(), True)])

        df_out = spark.createDataFrame(data = [dict(zip(["chi2_statistic", "p_value", "result"],
                                                        [chi2_stat, p_value, result]))], 
                                       schema = schema)
        
        return df_out