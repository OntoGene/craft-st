#!/bin/zsh


echo "Run shell script!\n"

# Set environment variables
echo "Set environment variables\n"



#! --  1. --
#! ---------------------------------- SET CONFIG -------------------------------

#? CUDA_VISIBLE_DEVICES
: "${cvd:=0}"

#? CONFIGURATION ids, global, bioes, pretrain, pretrained_ids 
: "${configuration:=ids}"

#? ONTOLOGY        CHEBI, CL, PR, GO_BP, GO_MF, GO_CC, MOP, NCBITaxon, PR, SO, UBERON...
: "${ontology:=CL_EXT}"

#? LABEL DETAIL ->  TAG SET SIZE
# for bioes , iob   -> BIOES, IOB
# for ids           -> [ignored]
# for pretraining   -> <tag-set size>, eg.: 1000
: "${label_detail:=BIOES}"

# Project root directory
: "${projdir:=data}"



#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
BIOBERT_DIR="$projdir/weights/biobert_v1.1_pubmed"



if [ $configuration = "global" ];then
    NER_DIR="$projdir/data/global"
    TMP_DIR="$projdir/tmp/global"
    LABEL_FORMAT='GLOBAL'

elif [ $configuration = "ids" ];then
    NER_DIR="$projdir/data/ids/${ontology}"
    TMP_DIR="$projdir/tmp/ids-${ontology}"

    LABEL_FORMAT=$ontology
    ONTOLOGY=$ontology

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method 

    LABEL_FORMAT="${ontology}.${label_detail}"
    NER_DIR="$projdir/data/pretrain/${LABEL_FORMAT}"
    TMP_DIR="$projdir/pretrained/${LABEL_FORMAT}"

    ONTOLOGY=$ontology

elif [ $configuration = "pretrained_ids" ];then
    LABEL_FORMAT="${ontology}.${label_detail}"
    NER_DIR="$projdir/data/ids/${ontology}"
    TMP_DIR="$projdir/tmp/pretrained-ids-${LABEL_FORMAT}"

    ONTOLOGY=$ontology

else
    NER_DIR="$projdir/data/spans/${ontology}"
    TMP_DIR="$projdir/tmp/spans-${ontology}"

    LABEL_FORMAT=$label_detail
    ONTOLOGY=$ontology
fi


export LABEL_FORMAT ONTOLOGY BIOBERT_DIR NER_DIR TMP_DIR

# Find the latest checkpoint.
checkpoint=$(ls $TMP_DIR/model.ckpt-*.data* | sort | tail -n 1)
checkpoint=${checkpoint%.data*}


#! -----------------------------------------------------------------------------
#! -----------------------------------------------------------------------------

# Test the model

printf "%-30s %s\n" "PRETRAINED MODEL:" $BIOBERT_DIR
printf "%-30s %s\n" "LABEL_FORMAT:" $LABEL_FORMAT
printf "%-30s %s\n" "ONTOLOGY:" $ONTOLOGY
printf "%-30s %s\n" "NER_DIR:" $NER_DIR
printf "%-30s %s\n" "TMP_DIR:" $TMP_DIR
printf "%-30s %s\n" "checkpoint:" $checkpoint
printf "%-30s %s\n" "CUDA_VISIBLE_DEVICES:" $cvd
printf "%-30s %s\n" "Configuration:" $configuration


sleep 6s


echo "Run prediction ...\n"

CUDA_VISIBLE_DEVICES=$cvd python run_ner_craft_bioes.py \
    --do_train=false \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$checkpoint \
    --data_dir=$NER_DIR/ \
    --output_dir=$TMP_DIR \
    --onto=$ONTOLOGY \
    --label_format=$LABEL_FORMAT \
    --configuration=$configuration



# Undo WordPiece tokenization
echo "Undo WordPiece tokenization ...\n"
CUDA_VISIBLE_DEVICES=$cvd python biocodes/ner_detokenize.py \
    --token_test_path=$TMP_DIR/token_test.txt \
    --label_test_path=$TMP_DIR/label_test.txt \
    --answer_path=$NER_DIR/test.tsv \
    --output_dir=$TMP_DIR
