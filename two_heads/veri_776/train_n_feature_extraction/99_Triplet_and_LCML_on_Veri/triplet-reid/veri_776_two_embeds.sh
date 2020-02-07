#!/bin/sh
#

# two losses 
EXP_ROOT="./veri_776_two_losses"

: '
# test
ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_test.csv"
SAVE_FILENAME='veri776_test_embeddings_two_losses_1.h5'

# query
#ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_query.csv"
#SAVE_FILENAME='veri776_query_embeddings_two_losses_1.h5'

python embed.py \
    --experiment_root $EXP_ROOT \
    --dataset $ANNO_ROOT \
    --filename $SAVE_FILENAME \
     "$@"
'

# test
#ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_test.csv"
#SAVE_FILENAME='veri776_test_embeddings_two_losses_2.h5'

# query
ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels_query.csv"
SAVE_FILENAME='veri776_query_embeddings_two_losses_2.h5'

python embed_2.py \
    --experiment_root $EXP_ROOT \
    --dataset $ANNO_ROOT \
    --filename $SAVE_FILENAME \
     "$@"
