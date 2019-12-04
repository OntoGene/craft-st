#!/bin/sh

#activate the right env
source deactivate
source activate tf_gpu


echo "Run shell script!\n"

# Set environment variables
echo "Set environment variables\n"

#! --  1. --
#! ---------------------------------- SET CONFIG -------------------------------

#? CHANGE MODEL NR TO SET THE RIGHT CHECKPOINT!!!
model_nr=204000

#? CUDA_VISIBLE_DEVICES
: "${cvd:=0}"

#? CONFIGURATION ids, global, bioes, pretrain, pretrained_ids 
: "${configuration:=ids}"

#? ONTOLOGY        CHEBI, CL, PR, GO_BP, GO_MF, GO_CC, MOP, NCBITaxon, PR, SO, UBERON...
: "${ontology:=CL_EXT}"

#? LABEL FORMAT ->  TAG SET SIZE
# for bioes , iob   -> BIOES, IOB
# for ids           -> CHEBI, CL, PR, ...
# for pretraining   -> <ontology>.<desiredSize> exp.: CHEBI.1000
: "${LABEL_FORMAT:=CL_EXT}"



#! --  2. --
#! --------------------------------- SET DIRECTORY -----------------------------

#! ++++++++++++++++++++++++++++++++++++++++
#! CHANGE IF YOU WANT TO LOAD INIT WEIGHTS
#! ++++++++++++++++++++++++++++++++++++++++
BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/weights/biobert_v1.1_pubmed'



if [ $configuration = "global" ];then
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_global_bert_data/fold0'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_global'
    LABEL_FORMAT='GOBAL'

elif [ $configuration = "ids" ];then

    # TMP_DIR='/home/user/jcornelius/tmpbert/bioner_craft_ids_CHEBI_whole'
    # BIOBERT_TEST_MODEL_DIR='/home/user/jcornelius/tmpbert/bioner_craft_ids_CHEBI_whole'

    # Normal + tag_set fold0 in run_ner_bioes.py
    # NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/whole'
    # TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/BUG_bioner_craft_ids_'$ontology'_whole'
    # BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ids_'$ontology'_whole'
    
    # EXT
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ext_ids_bert_data/'$ontology'/whole'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ext_ids_'$ontology'_whole'
    BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_ext_ids_'$ontology'_whole'
    
    
    ONTOLOGY=$ontology

elif [ $configuration = "pretrain" ];then
    #? is only valid in combination with ids
    #? otherwise rewrite the set_up_env() method 

    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll_train/'$LABEL_FORMAT
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT
    BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT

    ONTOLOGY=$ontology

elif [ $configuration = "pretrained_ids" ];then
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ids_bert_data/'$ontology'/fold0'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT
    BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_pretrained_ids_'$LABEL_FORMAT

    ONTOLOGY=$ontology

    BIOBERT_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/pretrained/'$LABEL_FORMAT

else
    # EXT
    NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_ext_bioes_bert_data/'$ontology'/whole'
    TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_ext_craft_bioes_finale_'$ontology
    BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_ext_craft_bioes_finale_'$ontology

    # not EXT
    # NER_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/data/craft_bioes_bert_data/'$ontology'/whole'
    # TMP_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_'$ontology
    # BIOBERT_TEST_MODEL_DIR='/mnt/storage/scratch1/jocorn/craft/biobert/tmp/bioner_craft_bioes_finale_'$ontology

    LABEL_FORMAT='BIOES'
    ONTOLOGY=$ontology
fi


export LABEL_FORMAT ONTOLOGY BIOBERT_DIR BIOBERT_TEST_MODEL_DIR NER_DIR TMP_DIR


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


sleep 6s

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