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

#? EPOCHS
epochs=20                

#? CUDA_VISIBLE_DEVICES
cvd=$1

#? CONFIGURATION ids, global, bioes, pretrain, pretrained_ids 
configuration="ids"

#? ONTOLOGY        CHEBI, CL, PR, GO_MF, SO , MOP, UBERON, GO_BP, NCBITaxon, GO_CC,  ...
ontology='GO_BP'

#? LABEL FORMAT ->  TAG SET SIZE
# for bioes , iob   -> BIOES, IOB
# for ids           -> CHEBI, CL, PR, GO_MF, SO ,UBERON, ...
# for pretraining   -> <ontology>.<desiredSize> exp.: CHEBI.1000
export LABEL_FORMAT='UBERON'



#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
export BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/weights/biobert_v1.1_pubmed'



if [ $configuration = "global" ];then
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_global_bert_data/fold0'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    export LABEL_FORMAT='GOBAL'

elif [ $configuration = "ids" ];then
    # fold0, whole
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/whole'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ids_'$ontology'_whole'
    
    #export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ext_ids_bert_data/'$ontology'/whole'
    #export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ext_ids_'$ontology'_whole'

    export ONTOLOGY=$ontology

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method 

    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll_train/'$LABEL_FORMAT'_full'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT
    
    export ONTOLOGY=$ontology

elif [ $configuration = "pretrained_ids" ];then
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/fold0'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT
    
    export ONTOLOGY=$ontology

    export BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT

else
    export NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_bioes_bert_data/'$ontology'/whole'
    export TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_'$ontology
    export LABEL_FORMAT='BIOES'
    export ONTOLOGY=$ontology
fi




#! -----------------------------------------------------------------------------
#! -----------------------------------------------------------------------------


printf "%-30s %s\n" "PRETRAINED MODEL:" $BIOBERT_DIR
printf "%-30s %s\n" "LABEL_FORMAT:" $LABEL_FORMAT
printf "%-30s %s\n" "ONTOLOGY:" $ONTOLOGY
printf "%-30s %s\n" "NER_DIR:" $NER_DIR
printf "%-30s %s\n" "TMP_DIR:" $TMP_DIR
printf "%-30s %s\n" "Epochs:" $epochs
printf "%-30s %s\n" "CUDA_VISIBLE_DEVICES:" $cvd
printf "%-30s %s\n" "Configuration:" $configuration

echo "\nTrain the model...\n"

sleep 8s


# Train the model

CUDA_VISIBLE_DEVICES=$cvd python run_ner_craft_bioes.py \
      --do_train=true \
      --do_eval=true \
      --num_train_epochs=$epochs \
      --vocab_file=$BIOBERT_DIR/vocab.txt \
      --bert_config_file=$BIOBERT_DIR/bert_config.json \
      --init_checkpoint=$BIOBERT_DIR/model.ckpt \
      --data_dir=$NER_DIR/ \
      --output_dir=$TMP_DIR \
      --onto=$ONTOLOGY \
      --label_format=$LABEL_FORMAT \
      --configuration=$configuration
