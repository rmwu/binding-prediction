# main script for training

######### training params
NUM_FOLDS=5
EPOCHS=100
SEED=0
CUDA=6

######### data params

# NOTE: change this for different models, e.g. "ligand" "bert" "structure"
TAG="bert"
CONFIG="config/${TAG}.json"

# NOTE: customize this to your data path
DATA="bindingdb"
ROOT="/data/rsg/chemistry/rmwu/data/processed/binding/${DATA}"

# NOTE: this specifies the dataset name, e.g. pdbbind.csv
SPLIT="bindingdb"
LABEL_PATH="${SPLIT}.csv"

# NOTE: you may customize the save directory's name here
if [ -z "$TAG" ]
then
    NAME="data=${DATA}_${SPLIT}-ep=${EPOCHS}"
else
    NAME="${TAG}-data=${DATA}_${SPLIT}-ep=${EPOCHS}"
fi

# NOTE: customize this to your save path
SAVE_PATH="/data/scratch/rmwu/tmp-runs/binding-predict/${NAME}"

echo $NAME

# NOTE: to use pre-trained model, uncomment --checkpoint_path and set
# accordingly
# NOTE: to run inference only, change --mode to "test"
python main.py \
    --mode "train" \
    --config_file $CONFIG \
    --data_file $LABEL_PATH \
    --run_name $NAME \
    --data_path $ROOT \
    --save_path $SAVE_PATH \
    --num_folds $NUM_FOLDS \
    --epochs $EPOCHS \
    --gpu $CUDA --seed $SEED
    #--checkpoint_path $SAVE_PATH \

