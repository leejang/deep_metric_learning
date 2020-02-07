#!/bin/sh
#

ANNO_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi/filenames_w_labels.csv"
IMAGE_ROOT="/workspace/04_test_cos_loss_on_Veri/VeRi"
INIT_CHECKPT="/workspace/04_test_cos_loss_on_Veri/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt"
EXP_ROOT="./veri_776_triplet_cosine"

python train.py \
    --train_set $ANNO_ROOT \
    --model_name mobilenet_v1_1_224 \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric cosine \
    --loss batch_hard \
    --learning_rate 3e-4 \
    --train_iterations 100000 \
    --decay_start_iteration 15000 \
    --checkpoint_frequency 10000 \
    "$@"
