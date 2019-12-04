#!/bin/sh


#activate the right env
source deactivate
source activate tf_gpu

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

#? LABEL FORMAT ->  TAG SET SIZE
# for bioes , iob   -> BIOES, IOB
# for ids           -> CHEBI, CL, PR, GO_MF, SO ,UBERON, ...
# for pretraining   -> <ontology>.<desiredSize> exp.: CHEBI.1000
: "${LABEL_FORMAT:=GO_CC_EXT}"



#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/weights/biobert_v1.1_pubmed'



if [ $configuration = "global" ];then
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_global_bert_data/fold0'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    LABEL_FORMAT='GOBAL'

elif [ $configuration = "ids" ];then
    # fold0, whole
    # NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/whole'
    # TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ids_'$ontology'_whole'
    
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ext_ids_bert_data/'$ontology'/whole'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ext_ids_'$ontology'_whole'

    ONTOLOGY=$ontology

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method 

    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll_train/'$LABEL_FORMAT'_full'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT
    
    ONTOLOGY=$ontology

elif [ $configuration = "pretrained_ids" ];then
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/fold0'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT
    
    ONTOLOGY=$ontology

    BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT

else
    # EXT
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ext_bioes_bert_data/'$ontology'/whole'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_ext_craft_bioes_finale_'$ontology

    # NOT ext
    # NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_bioes_bert_data/'$ontology'/whole'
    # TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_'$ontology
    
    LABEL_FORMAT='BIOES'
    ONTOLOGY=$ontology
fi


export LABEL_FORMAT ONTOLOGY BIOBERT_DIR NER_DIR TMP_DIR


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
