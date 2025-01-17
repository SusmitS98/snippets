import numpy as np
import pandas as pd
from sklearn.metrics import *
from pyspark.sql.types import *



def get_eval_df(raw_data, label_col, prediction_col, key_cols):

  def evaluate_classification_model(predictions):
  
      tot = len(predictions.index)
      positives = sum(predictions[label_col])
      predictions['AP'] = average_precision_score(predictions[label_col], predictions[prediction_col])
      fpr, tpr, _ = roc_curve(predictions[label_col], predictions[prediction_col])
      
      predictions['auc'] = auc(fpr, tpr)
      predictions['total'] = tot
      predictions['actual_positives'] = positives
      predictions['churn_ratio'] = positives/tot
      precision, recall, threshold_value = precision_recall_curve(predictions[label_col], predictions[prediction_col])
      precision = np.nan_to_num(precision)
      recall = np.nan_to_num(recall)
      f1_score= 2*precision*recall/(precision+recall)
      f1_score = np.nan_to_num(f1_score)
      ix = np.argmax(f1_score)
      predictions['threshold'] = threshold_value[ix]
      predictions['true_positives'] = (sum(predictions[predictions[prediction_col] > threshold_value[ix]][label_col]))
      predictions['pred_positives'] = len(predictions[predictions[prediction_col] > threshold_value[ix]])
      predictions['precision'] = precision[ix]
      predictions['recall'] = recall[ix]
      predictions['f1'] = f1_score[ix]
      
      try:   # This throws an error if any group has 0 positives.
        predictions['percentiles'] = pd.qcut(predictions[prediction_col].rank(method='first'), 100, labels = np.arange(1,101,1))
        
        predictions['tpr@10'] = (sum(predictions[predictions['percentiles'] > 90][label_col]))/ positives
        predictions['positives@10'] = (sum(predictions[predictions['percentiles'] > 90][label_col]))/(tot/10)
        predictions['total@10'] = (tot/10)
        
        predictions['tpr@20'] = (sum(predictions[predictions['percentiles'] > 80][label_col]))/ positives   
        predictions['positives@20'] = (sum(predictions[predictions['percentiles'] > 80][label_col]))/(tot/5)
        predictions['total@20'] = (tot/5)
        
        predictions['tpr@50'] = (sum(predictions[predictions['percentiles'] > 50][label_col]))/ positives
        predictions['positives@50'] = (sum(predictions[predictions['percentiles'] > 50][label_col]))/(tot/2)
        predictions['total@50'] = (tot/2)
      except:  
        predictions['tpr@10'] = 0
        predictions['positives@10'] = 0
        predictions['total@10'] = (tot/10)
        
        predictions['tpr@20'] = 0
        predictions['positives@20'] = 0
        predictions['total@20'] = (tot/5)
        
        predictions['tpr@50'] = 0
        predictions['positives@50'] = 0
        predictions['total@50'] = (tot/2)
    
      return predictions[key_cols + ['AP', 'auc',   'total', 'actual_positives','churn_ratio', 'threshold', 'true_positives', 'pred_positives', 'precision', 'recall', 'f1', 'tpr@10', 'positives@10','total@10', 'tpr@20', 'positives@20','total@20', 'tpr@50', 'positives@50','total@50']].tail(1)
  
  key_cols_struct = raw_data.select(key_cols).schema.fields
  schema = StructType(key_cols_struct + 
                      [StructField('AP', DoubleType(), True), StructField('auc', DoubleType(), True), StructField('total', DoubleType(), True), StructField('actual_positives', DoubleType(), True), StructField('churn_ratio', DoubleType(), True), StructField('threshold', DoubleType(), True), StructField('true_positives', DoubleType(), True), StructField('pred_positives', DoubleType(), True), StructField('precision', DoubleType(), True), StructField('recall', DoubleType(), True), StructField('f1', DoubleType(), True), StructField('tpr@10', DoubleType(), True), StructField('positives@10', DoubleType(), True), StructField('total@10', DoubleType(), True), StructField('tpr@20', DoubleType(), True), StructField('positives@20', DoubleType(), True), StructField('total@20', DoubleType(), True), StructField('tpr@50', DoubleType(), True), StructField('positives@50', DoubleType(), True), StructField('total@50', DoubleType(), True)])
  
  eval_df = raw_data.select([label_col, prediction_col] + key_cols).groupby(key_cols).applyInPandas(evaluate_classification_model, schema=schema)
  return eval_df