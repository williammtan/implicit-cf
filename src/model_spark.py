from pyspark.mllib.recommendation import ALS
from pyspark.sql import Row 

# TODO: Use pyspark mllib library

def train_pyspark_model(tr_interactions, hyperparams):
    # convert to interactions
    model = ALS.trainImplicit(tr_interactions, rank=hyperparams['latent_factors'], lambda_=hyperparams ['regularization'],  alpha=hyperparams['alpha'], iterations=hyperparams['iterations'])
    
    return model

