"""
MODEL.py - Does model stuff
"""

import os
import json
import math
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import normalize

import utils

import implicit

TEST_RATIO = 0.1
PURCHASE_PATH = '../data/u.purchase.csv'
SELECT_PATH1 = '../data/u.pr.select.csv'
SELECT_PATH2 = '../data/u.pa.select.csv'

######################################
### PREPROCESS
######################################
def preprocess(data_path):
    """
    Load dataset and create sparse train and test matrices.

    Args:
        - data_path: path to dataset
    Returns:
        - user_map: array of user IDs for each row of ratings matrix
        - item_map: array of user IDs for each column of ratings matrix
        - tr_sparse: sparse.csr_matrix training set
        - ts_sparse = sparse.csr_matrix test set
    """
    # get actual path and load csv to pd dataframe
    path = os.path.join(os.getcwd(), data_path)

    if data_path == SELECT_PATH1 or data_path == SELECT_PATH2:
        df = df_select(data_path)
    else:
        df = df_purchase(data_path)

    # index of user/item id is the row/col of rating matrix
    user_map = df.user_id.unique() # get unique users
    item_map = df.item_id.unique() # get unique items
    interactions = create_interactions(df, user_map, item_map)

    # split interactions to train and test
    tr_sparse, ts_sparse = train_test_sparse(interactions, test_ratio=TEST_RATIO, 
                                    user_map=user_map, item_map=item_map)

    return user_map, item_map, tr_sparse, ts_sparse

def df_purchase(path):
    """
    Convert purchase csv to DataFrame

    Args:
        - path: str to path
    
    Returns:
        - df: [user_id, item_id]
    """
    path = os.path.join(os.getcwd(), path)

    names = ['user_id','username','item_id','product_name','status']
    df = pd.read_csv(path, ',', names=names, engine='python', skiprows=1) # skip header row

    # drop unused columns and filter non successful transactions 
    df.drop(['username', 'product_name'], axis=1, inplace=True)
    df = df.drop(df[df.status!='SUCCESS'].index) # only keep SUCCESS status
    df.drop(['status'], axis=1, inplace=True) # df now only has user_id and item_id columns
    return df

def df_select(path):
    """
    Convert select csv to DataFrame

    Args:
        - path: str to path
    
    Returns:
        - df: [user_id, item_id]
    """
    path = os.path.join(os.getcwd(), path)

    names = ['user_id','user_pseudo_id','item_id','item_name']
    df = pd.read_csv(path, ',', names=names, engine='python', skiprows=1) # skip header row

    # drop unused columns 
    df.drop(['user_pseudo_id', 'item_name'], axis=1, inplace=True)

    return df


def create_interactions(df, user_map, item_map):
    """
    Generate array of user interactions 

    Args:
        - df: dataframe with user_id and item_id
    
    Returns:
        - interactions: array of user_interactions 
        in form: (user_index, item_index, num_of_interactions)
    """
    interactions = np.array([[0,0,0]]) # make temp interaction

    for u in user_map: # iterate users
        # count user u interaction with all items
        user_purchases = df[df.user_id==u].item_id # get all user u interactions with items
        unique, counts = np.unique(user_purchases, return_counts=True)
        u_interactions = np.array(list(zip(np.zeros(len(counts), dtype=float), unique, counts)))  # get count of interactions with items
        # u_interactions = [0, u, r_ui]
        for index, u_in in enumerate(u_interactions):
            # change user and item to indexes
            _, i, r_ui = u_in
            i_i = np.where(item_map==i)[0][0] # find item_id's index in item_map
            u_i = np.where(user_map==u)[0][0] # find user_id's index in the user_map
            u_interactions[index] = [u_i, i_i, int(r_ui)] # replace the u_interaction with indexes
        # add to interactions array
        interactions = np.vstack([interactions, u_interactions]) 

    np.delete(interactions, 0) # delete temp interaction
    return interactions

def train_test_sparse(interactions, test_ratio, user_map, item_map):
    """
    Splits the interactions list to train and test sparse matrices.

    Args:
        - interactions: array of user item interactions (u, i, r_ui)
        - test_ratio: ratio of test set
        - user_map: list of users
        - item_map: list of items
    
    Returns:
        - tr_sparse: sparse.csr_matrix of train interactions
        - ts_sparse: sparse.csr_matrix of test interactions
    """
    # shuffle interactions and split by test_ratio
    shuffled = np.random.permutation(interactions) # random order of interactions
    ts_length = round(test_ratio * len(interactions)) # round to avoid decimals

    ts_interactions = shuffled[:ts_length]
    tr_interactions = shuffled[ts_length:]

    # convert to sparse
    u_tr, i_tr, r_tr = zip(*tr_interactions) # get list of users, items and r_ui
    tr_sparse = sparse.csr_matrix((r_tr, (u_tr, i_tr)), 
                                shape=(len(user_map), len(item_map))) # make matrix of shape users*items

    u_ts, i_ts, r_ts = zip(*ts_interactions)
    ts_sparse = sparse.csr_matrix((r_ts, (u_ts, i_ts)), 
                                shape=(len(user_map), len(item_map)))
    
    np.save(os.path.join('../data/', 'user'), ts_sparse.toarray())

    return tr_sparse, ts_sparse

######################################
### MODEL
######################################

def train_model(tr_sparse, hyperparams):
    """
    Train model using Implicit library.

    Args:
        - tr_sparse: sparse.csr_matrix train
        - hyperparams:
            - latent_factors
            - regularization
            - iterations
            - alpha
    
    Returns:
        - user_factors: user feature matrix (X) 
        - item_factors: item feature matrix (Y)
    """

    model = implicit.als.AlternatingLeastSquares(factors=hyperparams['latent_factors'],  # make implicit als model
                    regularization=hyperparams['regularization'], iterations=hyperparams['iterations'])

    # Calculate confidence by multiplying by alpha
    data_conf = (tr_sparse.T * hyperparams['alpha']).astype('double')

    print('this can\'t print')
    model.fit(data_conf) # fit confidence matrix to model

    return model.user_factors, model.item_factors # return user features and item features matrix

def save_model(model_path, user_factors, item_factors, user_map, item_map):
    """
    Save model (user factors and item factors) in .npy file

    Args:
        - model_path: file path to .npy file
        - user_factors
        - item_factors
    """
    # create dir and npy files
    os.makedirs(model_path)
    np.save(os.path.join(model_path, 'user'), user_map)
    np.save(os.path.join(model_path, 'item'), item_map)
    np.save(os.path.join(model_path, 'row'), user_factors)
    np.save(os.path.join(model_path, 'col'), item_factors)

def rmse(user_factors, item_factors, ts_sparse, log=True):
    """
    Get rmse of prediction and test set

    Args:
        - user_factors
        - user_factors
        - ts_sparse
        - log: prints rmse (default to True)
    
    Returns:
        - rmse: root mean squared error
    """
    ts = ts_sparse.toarray()
    pred = np.dot(user_factors, item_factors.T) # get prediction matrix

    mse = 0
    count = 0
    for u in range(ts.shape[0]):
        # get sum of the squared difference in pred and ts
        for i in range(ts.shape[1]):
            if ts[u,i] != 0:
                mse += (pred[u,i] - ts[u,i])**2
                count +=1

    mse /= count # mean
    rmse = math.sqrt(mse)
    if log: print(f"RMSE: {rmse}")

    return rmse


######################################
### HYPER-PARAMETERS
######################################

def grid_search(tr_sparse, ts_sparse, config):
    """
    Grid search for best hyperparams.

    Args:
        - tr_sparse
        - ts_sparse
        - config[param]:
            - minValue
            - maxValue
            - scaleType: LINEAR or LOG
            - split
    
    Returns:
        - optimized_hyperparam
    """
    # split all params
    param_comb = {}
    for param, config in config.items():
        if config['scaleType'] == 'LINEAR':
            # split values
            split_dif = (config['maxValue'] - config['minValue']) / (config['split']-1)
            round_whole = param in ['latent_factors', 'iterations'] # only round when not latent factors of iterations
            comb = [(config['minValue'] + i*split_dif) for i in range(config['split'])] 
            comb = [round(val) for val in comb] if round_whole else [round(val, 1) for val in comb] # round if not dont_round
        elif config['scaleType'] == 'LOG':
            # convert values to log
            log_min = math.log10(config['maxValue'])
            log_max = math.log10(config['minValue'])
            log_dif = (log_min - log_max) / (config['split']-1)
            comb = [round(10**(log_max + i*log_dif), 3) for i in range(config['split'])] 
        param_comb[param] = comb
    # param_comb = {'latent_factors': [5, 18, 32, 45], 'alpha': [5.0, 16.7, 28.3, 40.0], 'regularization': [0.01, 0.037, 0.136, 0.5], 'iterations': [5, 18, 30]}

    # iterate params
    param_tests = []
    for latent_factors in param_comb['latent_factors']:
        for alpha in param_comb['alpha']:
            for regularization in param_comb['regularization']:
                for iterations in param_comb['iterations']:
                    # train model and get rmse
                    param_tests.append({'latent_factors': latent_factors, 'alpha': alpha, 
                                'regularization': regularization,'iterations': iterations})
    
    errors = np.array([])
    for i in tqdm(range(len(param_tests))):
        # train with params and get rmse
        hyperparams = param_tests[i]
        with utils.HiddenPrints(): # stop logging
            user_factors, item_factors = train_model(tr_sparse, hyperparams) # train model
            error = rmse(user_factors, item_factors, ts_sparse, log=False) # get rmse
            errors = np.append(errors, error)
    
    optimized_hyperparam = param_tests[np.argmax(errors)]
    print(f"OPTIMIZED PARAMETERS: {optimized_hyperparam}")

    return optimized_hyperparam


def random_search(tr_sparse, config):
    # TODO: Tune hyperparameters with random search
    pass


