#!/bin/sh

if [ -z "$1" ]
    then
        echo "You have to set the CUDA_VISIBLE_DEVICES\nFirst commandline-argument = CUDA_VISIBLE_DEVICES "
        return 0 
        #exit 1
fi
#activate the right env
source deactivate
source activate tf_gpu


echo "Run shell script!\n"

# Set enviorment Variables
echo "Set enviorment variables\n"

#! --  1. --
#! ---------------------------------- SET CONFIG -------------------------------

#? CHANGE MODEL NR TO SET THE RIGHT CHECKPOINT!!!
model_nr=110000
       
#? CUDA_VISIBLE_DEVICES
cvd=$1

#? CONFIGURATION ids, global, bioes, pretrain, pretrained_ids 
configuration="ids"

#? ONTOLOGY        CHEBI, CL, PR, GO_BP, ...
ontology='PR'

#? LABEL FORMAT ->  TAG SET SIZE
# for ids           -> CHEBI, CL, PR, ...
# for pretraining   -> <ontology>.<desiredSize> exp.: CHEBI.1000
export LABEL_FORMAT='PR'


#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
export BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/weights/biobert_v1.1_pubmed'



if [ $configuration = "global" ];then
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_global_bert_data/fold0'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    export BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    export LABEL_FORMAT='GOBAL'

elif [ $configuration = "ids" ];then
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/fold0'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ids_'$ontology'_fold0'
    export BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ids_'$ontology'_fold0'
    export ONTOLOGY=$ontology

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method 

    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll_train/'$LABEL_FORMAT
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT
    export BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT

    export ONTOLOGY=$ontology

elif [ $configuration = "pretrained_ids" ];then
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/fold0'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT
    export BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT

    export ONTOLOGY=$ontology

    export BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/PR.1000'

else
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_bioes_bert_data/GO_BP/whole'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_GO_BP'
    export BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_GO_BP'

    export LABEL_FORMAT='BIOES'
fi


#! -----------------------------------------------------------------------------
#! -----------------------------------------------------------------------------

# Test the model

printf "%-30s %s\n" "PRETRAINED MODEL:" $BIOBERT_DIR
printf "%-30s %s\n" "LABEL_FORMAT:" $LABEL_FORMAT
printf "%-30s %s\n" "ONTOLOGY:" $ONTOLOGY
printf "%-30s %s\n" "NER_DIR:" $NER_DIR
printf "%-30s %s\n" "TMP_DIR:" $TMP_DIR
printf "%-30s %s\n" "Model Nr.:" $model_nr
printf "%-30s %s\n" "CUDA_VISIBLE_DEVICES:" $cvd
printf "%-30s %s\n" "Configuration:" $configuration


sleep 5s

echo "Edit checkpoint file\n"
first_line='model_checkpoint_path: "model.ckpt-'$model_nr'"'
# edit first line of checkpoint file
sed -i "1s/.*/$first_line/" $BIOBERT_TEST_MODEL_DIR/checkpoint

echo "Run prediction ...\n"

CUDA_VISIBLE_DEVICES=$cvd python run_ner_craft_bioes.py \
    --do_train=false \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_TEST_MODEL_DIR/model.ckpt-$model_nr \
    --data_dir=$NER_DIR/ \
    --output_dir=$TMP_DIR \
    --onto=$ONTOLOGY \
    --label_format=$LABEL_FORMAT \
    --configuration=$configuration



# get entity-level evaluation
echo "Get entity level evaluation ...\n"
CUDA_VISIBLE_DEVICES=$cvd python biocodes/ner_detokenize.py \
    --token_test_path=$BIOBERT_TEST_MODEL_DIR/token_test.txt \
    --label_test_path=$BIOBERT_TEST_MODEL_DIR/label_test.txt \
    --answer_path=$NER_DIR/test.tsv \
    --output_dir=$TMP_DIR



# get entity-level evaluation
echo "Switch to python 2.7 enviroment to run conlleval.py\n"
source deactivate
source activate py27

echo "Run conlleval.py\n"
python conlleval.py $BIOBERT_TEST_MODEL_DIR/NER_result_conll.txt

source deactivate
source activate tf_gpu