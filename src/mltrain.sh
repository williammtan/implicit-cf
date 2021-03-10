# usage func
usage () {
    echo "usage: mltrain.sh [local | train | tune] [path_to/]<input_file>.csv [path_to/]<hyperparam_file/tune_config>.json

Use 'train' to create and train a model based on hyperparameters. Use 'tune' to optimize
hyperparameters.

Examples:

# train
./mltrain train ../data/u.purchase.csv /hyperparams/default.json

#tune
./mltrain tune ../data/u.purchase.csv /hyperparams/tune.config.json
"
}

# if all parameters are not filled
if [[ $# < 3 ]]; then
    usage
    exit 1
fi

date

TIME=`date +"%Y%m%d_%H%M%S"`

# set parameter values
TRAIN_JOB="$1" # train, tune, recommend, 
DATA_PATH="$2"
JOB_NAME=als_ml_${TRAIN_JOB}_${TIME}

if [[ ${TRAIN_JOB} == "train" ]]; then
    # assign path to model folder
    HYPERPARAM_PATH="$3"
    OUTPUT_PATH="../models/${JOB_NAME}"
    # mkdir -p $OUTPUT_PATH

    python3 task.py train $DATA_PATH $HYPERPARAM_PATH $OUTPUT_PATH # run train.py script
elif [[ ${TRAIN_JOB} == "tune" ]]; then
    HYPERPARAM_CONFIG_PATH="$3"
    OUTPUT_PATH="./hyperparams/${JOB_NAME}"

    python3 task.py tune $DATA_PATH $HYPERPARAM_CONFIG_PATH $OUTPUT_PATH # run tune.py script
else
    usage
fi

