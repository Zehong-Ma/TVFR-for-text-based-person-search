#!/bin/bash
#GPU_ID=$1,$2,$3,$4
export CUDA_VISIBLE_DEVICES=1
# Where the dataset is saved to.
DATASET_DIR=/home/B/tanth/datasets/CUHK-PEDES/TFRecords/pedes
RESTORE_PATH=/home/B/tanth/pretrain_model/mobilenet_v1/mobilenet_v1_1.0_224.ckpt

DATASET_NAME=pedes
SAVE_NAME=pedes_mobilenet_with_6_parts_with_part_loss_share_weight0.1
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs
SAMPLE_DIR=${SAVE_NAME}/train_samples

SPLIT_NAME=train

# Model setting
MODEL_NAME=mobilenet_v1
MODEL_SCOPE=MobilenetV1
RESTORE_SCOPES=MobilenetV1
EXCLUDE_SCOPES=MobilenetV1/Logits



# Run training.
python train_image_text.py \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --train_samples_dir=${SAMPLE_DIR} \
    --dataset_name=${DATASET_NAME}\
    --dataset_split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --model_scope=${MODEL_SCOPE} \
    --preprocessing_name=${MODEL_NAME} \
    --restore_pretrain=True \
    --restore_path=${RESTORE_PATH} \
    --restore_scopes=${RESTORE_SCOPES} \
    --checkpoint_exclude_scopes=${EXCLUDE_SCOPES} \
    --num_epochs=50 \
    --ckpt_steps=5000 \
    --batch_size=16 \
    --group_size=6 \
    --image_height=384 \
    --image_width=128 \
    --num_classes=11004 \
    --optimizer=adam \
    --learning_rate=0.0002 \
    --opt_epsilon=1e-8 \
    --CMPM=True \
    --CMPC=True \
    --IMAGE_PART_LOSS=True \
    --IMAGE_PART_LOSS_weight=0.1 \
    --IMAGE_PART_LOSS_before_multiply_text_weight=True
