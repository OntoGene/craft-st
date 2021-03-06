#!/bin/zsh


echo "Run shell script!\n"

# Set environment variables
echo "Set environment variables\n"



#! --  1. --
#! ---------------------------------- SET CONFIG -------------------------------

#? EPOCHS
: "${epochs:=55}"

#? CUDA_VISIBLE_DEVICES
: "${cvd:=0}"

#? CONFIGURATION ids, global, bioes, pretrain, pretrained_ids 
: "${configuration:=ids}"

#? ONTOLOGY        CHEBI, CL, PR, GO_MF, SO , MOP, UBERON, GO_BP, NCBITaxon, GO_CC,  ...
: "${ontology:=GO_CC_EXT}"

#? LABEL DETAIL ->  TAG SET SIZE
# for bioes , iob   -> BIOES, IOB
# for ids           -> [ignored]
# for pretraining   -> <tag-set size>, eg.: 1000
: "${label_detail:=BIOES}"

# Run prediction on test set right after training?
: "${DO_PREDICT:=true}"

# Project root directory
: "${projdir:=data}"



#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
BIOBERT_DIR="$projdir/weights/biobert_v1.1_pubmed"
checkpoint="$BIOBERT_DIR/model.ckpt"


if [ $configuration = "global" ];then
    NER_DIR="$projdir/data/global"
    TMP_DIR="$projdir/tmp/global"

elif [ $configuration = "ids" ];then
    NER_DIR="$projdir/data/ids/${ontology}"
    TMP_DIR="$projdir/tmp/ids-${ontology}"

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method

    NER_DIR="$projdir/data/pretrain/${ontology}.${label_detail}"
    TMP_DIR="$projdir/pretrained/${ontology}.${label_detail}"

    DO_PREDICT=false

elif [ $configuration = "pretrained_ids" ];then
    NER_DIR="$projdir/data/ids/${ontology}"
    TMP_DIR="$projdir/tmp/pretrained-ids-${ontology}.${label_detail}"

    pretrained="$projdir/pretrained/${ontology}.${label_detail}"
    checkpoint=$(ls "$pretrained"/model.ckpt-*.data* | sort | tail -n 1)
    checkpoint=${checkpoint%.data*}

else
    NER_DIR="$projdir/data/spans/${ontology}"
    TMP_DIR="$projdir/tmp/spans-${ontology}"
fi


#! -----------------------------------------------------------------------------
#! -----------------------------------------------------------------------------


printf "%-30s %s\n" "PRETRAINED MODEL:" $BIOBERT_DIR
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
      --do_predict=$DO_PREDICT \
      --num_train_epochs=$epochs \
      --vocab_file=$BIOBERT_DIR/vocab.txt \
      --bert_config_file=$BIOBERT_DIR/bert_config.json \
      --init_checkpoint=$checkpoint \
      --data_dir=$NER_DIR/ \
      --output_dir=$TMP_DIR \
      --configuration=$configuration
