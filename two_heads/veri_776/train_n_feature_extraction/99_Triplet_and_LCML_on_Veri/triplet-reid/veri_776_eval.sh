#!/bin/sh
#

# triplet with BH
#EXP_ROOT="./veri_776"
#QUERY_FILENAME='veri776_query_embeddings.h5'
#TEST_FILENAME='veri776_test_embeddings.h5'
#DIST_METRIC='euclidean'
#SAVE_FILENAME='veri_776_evaluation.json'

# lmcl without BH
#EXP_ROOT="./veri_776_lmcl"
#QUERY_FILENAME='veri776_query_embeddings_lmcl.h5'
#TEST_FILENAME='veri776_test_embeddings_lmcl.h5'
#DIST_METRIC='cosine'
#SAVE_FILENAME='veri_776_evaluation.json'

# lmcl with BH
#EXP_ROOT="./veri_776_lmcl_bh"
#QUERY_FILENAME='veri776_query_embeddings_lmcl_bh.h5'
#TEST_FILENAME='veri776_test_embeddings_lmcl_bh.h5'
#DIST_METRIC='cosine'
#SAVE_FILENAME='veri_776_evaluation.json'

# two losses (triplet and lmcl) - use triplet emb only (1)
#EXP_ROOT="./veri_776_two_losses"
#QUERY_FILENAME='veri776_query_embeddings_two_losses_1.h5'
#TEST_FILENAME='veri776_test_embeddings_two_losses_1.h5'
#DIST_METRIC='cosine'
#SAVE_FILENAME='veri_776_evaluation_two_losses_1.json'

# two losses (triplet and lmcl) - use lmcl emb only (2)
#EXP_ROOT="./veri_776_two_losses"
#QUERY_FILENAME='veri776_query_embeddings_two_losses_2.h5'
#TEST_FILENAME='veri776_test_embeddings_two_losses_2.h5'
#DIST_METRIC='cosine'
#SAVE_FILENAME='veri_776_evaluation_two_losses_2.json'

: '
python evaluate.py \
    --excluder veri_776 \
    --query_dataset /workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_pids_query.csv \
    --query_embeddings $EXP_ROOT/$QUERY_FILENAME \
    --gallery_dataset /workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_pids_test.csv \
    --gallery_embeddings $EXP_ROOT/$TEST_FILENAME \
    --metric $DIST_METRIC \
    --filename $EXP_ROOT/$SAVE_FILENAME
'

# two losses (triplet and lmcl) - use the both embs
EXP_ROOT="./veri_776_two_losses"
QUERY_FILENAME_1='veri776_query_embeddings_two_losses_1.h5'
TEST_FILENAME_1='veri776_test_embeddings_two_losses_1.h5'
QUERY_FILENAME_2='veri776_query_embeddings_two_losses_2.h5'
TEST_FILENAME_2='veri776_test_embeddings_two_losses_2.h5'
DIST_METRIC_1='euclidean'
DIST_METRIC_2='cosine'
SAVE_FILENAME='veri_776_evaluation_two_losses_both_together.json'

python evaluate_2.py \
    --excluder veri_776 \
    --query_dataset /workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_pids_query.csv \
    --query_embeddings $EXP_ROOT/$QUERY_FILENAME_1 \
    --query_embeddings_2 $EXP_ROOT/$QUERY_FILENAME_2 \
    --gallery_dataset /workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_pids_test.csv \
    --gallery_embeddings $EXP_ROOT/$TEST_FILENAME_1 \
    --gallery_embeddings_2 $EXP_ROOT/$TEST_FILENAME_2 \
    --metric $DIST_METRIC_1 \
    --metric_2 $DIST_METRIC_2 \
    --filename $EXP_ROOT/$SAVE_FILENAME


