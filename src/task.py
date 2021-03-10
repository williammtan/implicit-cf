"""
Train/tune model with set of hyperparams

train():
    Input: python3 task.py {DATA_PATH} {HYPERPARAM_PATH} {OUTPUT_PATH}

tune():
    Input: python3 task.py {DATA_PATH} {HYPERPARAM_CONFIG_PATH} {OUTPUT_PATH}
"""
print("Importing modules...")

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import os

import model
import utils

def train():
    # get params past from mltrain.sh
    data_path = sys.argv[2]
    hyperparam_path = sys.argv[3]
    output_path = sys.argv[4]

    # preprocess data
    user_map, item_map, tr_sparse, ts_sparse = model.preprocess(data_path)

    # run model
    hyperparams = utils.read_json(hyperparam_path)
    user_factors, item_factors = model.train_model(tr_sparse, hyperparams)

    # save model
    model.save_model(output_path, user_factors, item_factors,  user_map, item_map)

    # log metrics
    rmse = model.rmse(user_factors, item_factors, ts_sparse)

def tune():
    # get params past from mltrain.sh
    data_path = sys.argv[2]
    config_path = sys.argv[3]
    output_path = sys.argv[4]

    # preprocess data
    user_map, item_map, tr_sparse, ts_sparse = model.preprocess(data_path)

    config_data = utils.read_json(config_path) # load config hyperparams
    tune_method = config_data['method']

    if tune_method == 'random-search':
        # random search
        pass
    elif tune_method == 'grid-search':
        # grid search
        optimized_param = model.grid_search(tr_sparse, ts_sparse, config_data['params'])
    
    # save params
    utils.write_json(output_path+'.json', optimized_param)


if __name__ == '__main__':
    print(f'Starting {sys.argv[1]}...')
    if sys.argv[1] == 'train':
        train()
    else:
        tune()