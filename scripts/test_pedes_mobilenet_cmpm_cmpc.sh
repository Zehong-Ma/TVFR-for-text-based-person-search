#!/bin/bash
#
export CUDA_VISIBLE_DEVICES=2
# Where the dataset is saved to.
DATASET_DIR=/home/B/tanth/datasets/CUHK-PEDES/TFRecords/pedes


# Where the checkpoint and logs saved to.
DATASET_NAME=pedes
SAVE_NAME=pedes_mobilenet_with_6_parts_with_part_loss_share_weight0.1
CKPT_DIR=${SAVE_NAME}/checkpoint
RESULT_DIR=${SAVE_NAME}/results

MODEL_NAME=mobilenet_v1
MODEL_SCOPE=MobilenetV1

SPLIT_NAME=test

for i in $(seq 2 30)
do
    # Run evaluation.
    python test_image_text.py \
      --checkpoint_dir=${CKPT_DIR} \
      --eval_dir=${RESULT_DIR} \
      --dataset_name=${DATASET_NAME} \
      --split_name=${SPLIT_NAME} \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL_NAME} \
      --model_scope=${MODEL_SCOPE} \
      --preprocessing_name=${MODEL_NAME} \
      --group_size=6 \
      --batch_size=1 \
      --ckpt_num=${i}

    echo "Evaluating with Cosine Distance..."
    python2 evaluation/bidirectional_eval.py ${RESULT_DIR} --method cosine

    # echo "Waiting..."
    # sleep 16m

done
