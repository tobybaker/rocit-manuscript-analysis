import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt

import itertools
import polars as pl
import datahelper_xgboost

from pathlib import Path

def generate_xgboost_param_sweep(param_grid):

    keys = param_grid.keys()
    values = param_grid.values()
    # Create cartesian product of all parameter values
    combinations = itertools.product(*values)
    # Convert each combination into a dictionary
    param_dicts = [dict(zip(keys, combo)) for combo in combinations]

    return param_dicts

def get_param_grid():
    
    param_grid = {
    'max_depth': [5,7,9,11],
    'learning_rate': [0.01],
    'num_boost_round':[15000],
    'subsample':[0.25,0.5,0.6,0.8,1.0],
    'min_child_weight':[3.0,5.0,10.0,25.0],
    'colsample_bytree':[0.25,0.5,0.75,1.0],
    'early_stopping_fraction':[0.1],
    'scale_pos_weight':[1.0],
    'objective': ['binary:logistic'],
    'eval_metric': ['auc'],
    'tree_method':['hist']}

    return param_grid



def get_run_metrics(pred_probs,sample_data):
    predictions = (pred_probs > 0.5).astype(int)  # Convert probabilities to binary labels
    run_metrics = {}
    # Evaluate model
    run_metrics['accuracy'] = accuracy_score(sample_data['y_test'], predictions)
    run_metrics['auc'] = roc_auc_score(sample_data['y_test'], pred_probs)
    run_metrics['precision'] = precision_score(sample_data['y_test'], predictions)
    run_metrics['recall'] = recall_score(sample_data['y_test'], predictions)
    run_metrics['f1'] = f1_score(sample_data['y_test'], predictions)
    run_metrics['mcc'] = matthews_corrcoef(sample_data['y_test'], predictions)
    run_metrics['proportion_positive'] = np.mean(sample_data['y_test'])
    return run_metrics

def get_xgb_matrix(sample_data):
    dtrain = xgb.DMatrix(sample_data['X_train'], label=sample_data['y_train'])
    dval = xgb.DMatrix(sample_data['X_val'], label=sample_data['y_val'])
    dtest= xgb.DMatrix(sample_data['X_test'], label=sample_data['y_test'])
    return dtrain,dval,dtest

def run_model_grid_search(sample_data,sample_id):

    param_grid = get_param_grid()
    param_store = generate_xgboost_param_sweep(param_grid)

    best_auc = -1
    best_run_data = None

    for params in param_store:
        params = dict(params)
        params['scale_pos_weight']= np.mean(1-sample_data['y_train'])/np.mean(sample_data['y_train'])
        run_data = {'sample_id':sample_id}
        run_data.update(params)
        
        num_round = params.pop('num_boost_round')
        early_stopping_rounds = int(num_round*params.pop('early_stopping_fraction'))

        dtrain,dval,dtest = get_xgb_matrix(sample_data)
        evals = [(dtrain, 'train'), (dval, 'val')]

        bst = xgb.train(params, dtrain, num_round,early_stopping_rounds=early_stopping_rounds,evals=evals,verbose_eval=False)

        pred_probs = bst.predict(dtest)
        run_metrics = get_run_metrics(pred_probs,sample_data)
        run_data.update(run_metrics)
        run_data['n_rounds_used'] = bst.best_iteration + 1

        if run_data['auc'] > best_auc:
            best_auc = run_data[f'auc']
            best_run_data = run_data

    return pl.DataFrame(best_run_data)
            


if __name__ =='__main__':
    
    all_sample_ids = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    sample_id = all_sample_ids[int(sys.argv[1])]
 
    sample_data = datahelper_xgboost.get_training_data_dict(sample_id)
    sample_out_store = run_model_grid_search(sample_data,sample_id)
    
    out_dir= Path('/hot/user/tobybaker/ROCIT_Paper/predictions/xgboost')
    out_path = out_dir/f'{sample_id}_xgboost_results.tsv'
    sample_out_store.write_csv(out_path,separator='\t')
    print(sample_out_store)

    
  