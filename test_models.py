"""
    Generic evaluation script that evaluates a model using a given dataset.
    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import json
import itertools
import collections
from datetime import datetime
import numpy as np
import os.path
import sys
import scipy.io as sio
from collections import defaultdict

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import *
from modules import *

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def merge_image_features(all_fc_layers, all_part_fc_layers, all_labels, all_filenames):
    """
        Merge features from the same image.
    """
    merged = list(itertools.chain.from_iterable(all_filenames))
    with open(os.path.join(FLAGS.eval_dir, 'filenames.txt'), 'w') as myfile:
        myfile.write('\n'.join(merged))
    myfile.close()
    dup_list = sorted(list_duplicates(merged), key=lambda x: x[1])
    num_images = len(dup_list)
    avg_fc_layers = []
    avg_labels = []
    avg_part_fc_layers = []
    
    for i in range(num_images):
        single_avg_layer = []
        single_part_avg_layer = []

        for j in range(len(all_fc_layers[0])): ## 0~group size
            single_fc_layers = [all_fc_layers[k][j] for k in dup_list[i][1]]
            single_avg_layer.append(sum(single_fc_layers)/len(single_fc_layers)) # len: group_size
        
        for jj in range(len(all_part_fc_layers[0])):
            single_part_fc_layers = [all_part_fc_layers[k][jj] for k in dup_list[i][1]]
            single_part_avg_layer.append(sum(single_part_fc_layers)/len(single_part_fc_layers))

        single_labels = [all_labels[k] for k in dup_list[i][1]]

        avg_fc_layers.append(single_avg_layer)
        avg_labels.append(sum(single_labels) / len(single_labels))
        avg_part_fc_layers.append(single_part_avg_layer)
    
    # avg_fc_layers: num_images * group_size * 512

    return avg_fc_layers, avg_part_fc_layers, avg_labels



def save_array(features, name):
    np_features = np.asarray(features)
    if name =='image':
        np_features = np.reshape(np_features, [len(features), -1])
        feature_filename = "%s/%s_%s_features.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
        np.save(feature_filename, np_features)
    if name == 'image_part':
        np_features = np.reshape(np_features,[np_features.shape[0],np_features.shape[1],-1])
        feature_filename = "%s/%s_%s_features.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
        np.save(feature_filename, np_features)
    
    elif name =='caption':
        np_features = np.reshape(np_features, [len(features), -1])
        feature_filename = "%s/%s_%s_features.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
        np.save(feature_filename, np_features)

    elif name == 'image_labels' or name == 'caption_labels':
        np_labels = np.reshape(np_features, len(features))
        label_filename = "%s/%s_%s.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
        np.save(label_filename, np_labels)
    
    else: #  coefficient
        feature_filename = "%s/%s_%s.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
        np.save(feature_filename, features)

    # print('save .npy successfully in %s/%s_%s '%(FLAGS.eval_dir, FLAGS.split_name, name))
    # save .mat
    # feature_filename = "%s/%s_%s_features.mat" % (FLAGS.eval_dir, FLAGS.split_name, name)
    # sio.savemat(feature_filename, {'feature': np_features})
    # label_filename = "%s/%s_%s_labels.mat" % (FLAGS.eval_dir, FLAGS.split_name, name)
    # sio.savemat(label_filename, {'label': np_labels})


# def save_features(fc_layers, text_coefficient, part_fc_layers, text_coefficient_for_part, caption_embeddings, labels, filenames, images, num_examples, saver):
def save_features(image_base_embeddings, part_fc_layers, text_coefficient_for_part, caption_embeddings, labels, filenames, images, num_examples, saver):

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            ckpt_name = ckpt.all_model_checkpoint_paths[FLAGS.ckpt_num]
            global_step = ckpt_name.split('/')[-1].split('-')[-1]
            if os.path.exists(ckpt_name+'.index'):
                saver.restore(sess, ckpt_name)
                print('Succesfully loaded model from %s at step=%s.' %
                      (ckpt_name, global_step))

            else:
                print('No checkpoint file found')
                return

            if FLAGS.max_num_batches:
                num_batches = FLAGS.max_num_batches
            else:
                num_batches = int(math.ceil(num_examples / float(FLAGS.batch_size)))
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
            
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                step = 0
                all_image_embeddings, all_caption_features, all_labels, all_filenames =  [], [], [], []
                all_part_fc_layers, all_text_coefficient_for_part = [], []
                print("Current Path: %s" % os.getcwd())
                print('%s: starting extracting features on (%s).' % (datetime.now(), FLAGS.split_name))
                while step < num_batches and not coord.should_stop():
                    step += 1
                    if step % 1000 == 0:
                        sys.stdout.write('\r>> Extracting %s image %d/%d [%d examples]' %
                                    (FLAGS.split_name, step, num_batches, num_examples))
                        sys.stdout.flush()
                    eval_image_embeddings, eval_part_fc_layers, eval_text_coefficient_for_part, \
                    eval_caption_embeddings, eval_labels, eval_filenames = sess.run(
                        [image_base_embeddings, part_fc_layers, text_coefficient_for_part, caption_embeddings, labels, filenames])
                    # pdb.set_trace()
                    # eval_fc_layers = np.squeeze(np.array(eval_fc_layers),axis=1)
                    eval_image_embeddings = np.reshape(eval_image_embeddings, [eval_image_embeddings.shape[0], -1])
                    eval_caption_features = np.reshape(eval_caption_embeddings, [eval_caption_embeddings.shape[0], -1])
                    # eval_text_coefficient = np.squeeze(np.array(eval_text_coefficient), axis=0)
                    
                    eval_part_fc_layers = np.squeeze(np.array(eval_part_fc_layers), axis=1)
                    eval_text_coefficient_for_part = np.squeeze(np.array(eval_text_coefficient_for_part), axis=0)
                    # pdb.set_trace()

                    all_image_embeddings.append(eval_image_embeddings)
                    # all_text_coefficient.append(eval_text_coefficient)

                    all_part_fc_layers.append(eval_part_fc_layers)
                    all_text_coefficient_for_part.append(eval_text_coefficient_for_part)

                    all_caption_features.append(eval_caption_features)
                    all_labels.append(eval_labels)
                    all_filenames.append(eval_filenames)
                    
                #  save features and labels
                avg_image_embeddings, avg_part_fc_layers, avg_labels = merge_image_features(all_image_embeddings, all_part_fc_layers, all_labels, all_filenames)

                # save_array(avg_fc_layers, 'image')
                save_array(avg_image_embeddings, 'image')
                save_array(avg_labels, 'image_labels')
                save_array(all_caption_features, 'caption')
                save_array(all_labels, 'caption_labels')
                save_array(avg_part_fc_layers, 'image_part')
                # save_array(all_text_coefficient,'coefficient')
                save_array(all_filenames, 'filename')
                save_array(all_text_coefficient_for_part,'coefficient_part')
                print("Done!")

            except Exception as e:
                # print(22222)
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
def evaluate():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=None,
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label, caption_id, caption, filename] = \
            provider.get(['image', 'label', 'caption_ids', 'caption', 'filename'])

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        #eval_image_size = network_fn.default_image_size

        image = image_preprocessing_fn(image, FLAGS.image_height, FLAGS.image_width)

        caption_length = tf.shape(caption_id)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
        input_seq = tf.slice(caption_id, [0], input_length)
        target_seq = tf.slice(caption_id, [1], input_length)
        input_mask = tf.ones(input_length, dtype=tf.int32)

        images, labels, input_seqs, target_seqs, input_masks, captions, caption_ids, filenames = \
            tf.train.batch([image, label, input_seq, target_seq, input_mask, caption, caption_id, filename],
                           batch_size=FLAGS.batch_size,
                           num_threads=FLAGS.num_preprocessing_threads,
                           capacity=5 * FLAGS.batch_size,
                           dynamic_pad=True)

        ####################
        # Define the model #
        ####################
        image_features, _, image_features_3d = build_image_features(network_fn, images)
        text_features, _ = build_text_features(input_seqs, input_masks)
        
        
        image_base_embeddings = build_joint_embeddings(image_features, scope='image_joint_embedding')
        text_embeddings = build_joint_embeddings(text_features, scope='text_joint_embedding')

        part_fc_layers = build_part_conv_features(image_features_3d)
        if FLAGS.random_filter == 'norm':
            text_coefficient_for_part = generate_normal_random_filter()
        elif FLAGS.random_filter == 'uniform':
            text_coefficient_for_part = generate_uniform_random_filter()
        else:
            text_coefficient_for_part = build_text_coefficient_for_part(text_features, FLAGS.group_size)

        # text_coefficient_for_part = build_text_coefficient_for_part(text_features, FLAGS.group_size)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # fc_layers, text_fc_coefficient,
        save_features(image_base_embeddings, part_fc_layers, text_coefficient_for_part, text_embeddings, \
            labels, filenames, images, dataset.num_samples, saver)
        



