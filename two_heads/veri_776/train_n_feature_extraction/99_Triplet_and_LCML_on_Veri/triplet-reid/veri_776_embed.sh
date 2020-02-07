#!/bin/sh
#

# triplet with BH
#EXP_ROOT="./veri_776"

# lmcl without BH
EXP_ROOT="./veri_776_lmcl"

# lmcl with BH
#EXP_ROOT="./veri_776_lmcl_bh"

# triplet cosine 
#EXP_ROOT="./veri_776_triplet_cosine"

# test
ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_test.csv"
#SAVE_FILENAME='veri776_test_embeddings.h5'
SAVE_FILENAME='veri776_test_embeddings_lmcl.h5'
#SAVE_FILENAME='veri776_test_embeddings_lmcl_bh.h5'
#SAVE_FILENAME='veri776_test_embeddings_triplet_cosine.h5'

# query
#ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_query.csv"
#SAVE_FILENAME='veri776_query_embeddings.h5'
#SAVE_FILENAME='veri776_query_embeddings_lmcl.h5'
#SAVE_FILENAME='veri776_query_embeddings_lmcl_bh.h5'
#SAVE_FILENAME='veri776_query_embeddings_triplet_cosine.h5'

python embed.py \
    --experiment_root $EXP_ROOT \
    --dataset $ANNO_ROOT \
    --filename $SAVE_FILENAME \
     "$@"
