import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
#from sklearn.metrics.pairwise import pairwise_distances
#from sklearn import preprocessing
import os.path as osp
import math
import sys
import time

from retrieval_eval import *
from datetime import datetime


def _get_test_data(result_dir):
    all_captions = np.load(osp.join(result_dir, 'test_caption_features.npy'))
    all_labels = np.load(osp.join(result_dir, 'test_caption_labels.npy'))
    all_avg_fc_layers = np.load(osp.join(result_dir, 'test_image_features.npy'))
    all_avg_labels = np.load(osp.join(result_dir, 'test_image_labels.npy'))
    # all_coefficient = np.load(osp.join(result_dir, 'test_coefficient_features.npy'))

    all_avg_fc_layers_for_part = np.load(osp.join(result_dir, 'test_image_part_features.npy'))
    all_coefficient_for_part = np.load(osp.join(result_dir, 'test_coefficient_part.npy'))

    return all_captions, all_labels, all_avg_fc_layers, all_avg_labels, all_avg_fc_layers_for_part, all_coefficient_for_part


def _eval_retrieval(PY, GY, D):

    # D_{i, j} is the distance between the ith array from PX and the jth array from GX.
    
    
    Rank = np.argsort(D, axis=1)
    np.save("search_mat.npy",np.array(Rank))
    # Evaluation
    recall_1 = recall_at_k(Rank, PY, GY, k=1)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@1', recall_1)

    recall_5 = recall_at_k(Rank, PY, GY, k=5)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@5', recall_5)

    recall_10 = recall_at_k(Rank, PY, GY, k=10)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@10', recall_10)

    recall_sum = recall_1 + recall_5 + recall_10
    print "{:8}{:8.2%}".format('Recallsum', recall_sum)

    map_value = 0.0
    # map_value = mean_average_precision(Rank, PY, GY)  # Mean Average Precision
    # print "{:8}{:8.2%}".format('MAP', map_value)

    return recall_1, recall_5, recall_10, recall_sum, map_value

def cosine_pairwise_distance(x1,x2, batch_size, num_avg_samples):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1))
    
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1)) 
      
    x1_x2 = tf.reduce_sum(tf.einsum('bf,baf->baf',x1, x2), axis=-1)
    
    cosine = tf.divide(x1_x2, tf.einsum('b,ba->ba',x1_norm, x2_norm))

    one_tensor = tf.constant(1,dtype=tf.float32,shape=[batch_size,num_avg_samples])

    distance = tf.subtract(one_tensor,cosine)

    return distance

def main(args):
    all_captions , all_labels, all_avg_fc_layers, all_avg_labels, \
        all_avg_fc_layers_for_part, all_coefficient_for_part = _get_test_data(args.result_dir)
    print("load varables successfully")
    
    feature_size = 512 
    batch_size= 18
    num_examples = all_captions.shape[0]
    num_avg_samples = all_avg_fc_layers.shape[0]
    
    # all_coefficient = tf.convert_to_tensor(all_coefficient,dtype=tf.float32)   
    all_avg_fc_layers = tf.convert_to_tensor(all_avg_fc_layers,dtype=tf.float32)  
    all_captions  = tf.convert_to_tensor(all_captions,tf.float32)  

    all_coefficient_for_part = tf.convert_to_tensor(all_coefficient_for_part, dtype=tf.float32)   
    all_avg_fc_layers_for_part = tf.convert_to_tensor(all_avg_fc_layers_for_part, dtype=tf.float32)  

    num_batches = int(math.ceil(num_examples / float(batch_size)))

    D = []

    config = tf.ConfigProto(
            allow_soft_placement=True,
            )    
    config.gpu_options.allow_growth = True
    
    input_queue = tf.train.slice_input_producer([all_captions, all_coefficient_for_part],num_epochs=num_batches,shuffle=False)

    captions, coefficients_part = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    
    # image_dcs_whole = tf.reduce_sum(tf.einsum('bg,agf->bagf', coefficients, all_avg_fc_layers), axis=2)
    image_dcs_part = tf.reduce_sum(tf.einsum('bg,agf->bagf', coefficients_part, all_avg_fc_layers_for_part), axis=2)

    image_dcs = all_avg_fc_layers  + image_dcs_part 
    
    distances = cosine_pairwise_distance(captions, image_dcs, batch_size, num_avg_samples)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(num_batches):
            #sys.stdout.write('\r>> Testing image %d/%d [%d examples]' %
            #                        (i+1, num_batches, num_examples))
            #sys.stdout.flush()

            distance_result = sess.run(distances)

            D.extend(distance_result)
            
    
    mytime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    text_file = open(osp.join(args.result_dir, "retrieval-%s.txt" % mytime), "w")
    print("Text-to-Image Evaluation...")
    recall_1, recall_5, recall_10, recall_sum, map_value = _eval_retrieval(all_labels, all_avg_labels, D)
    text_file.write("Text-to-Image Evaluation \n")
    text_file.write("{:8}{:8.2%}\n".format('Recall@1', recall_1))
    text_file.write("{:8}{:8.2%}\n".format('Recall@5', recall_5))
    text_file.write("{:8}{:8.2%}\n".format('Recall@10', recall_10))
    text_file.write("{:8}{:8.2%}\n".format('Recallsum', recall_sum))
    text_file.write("{:8}{:8.2%}\n".format('MAP', map_value))

    """
    print("Image-to-Text Evaluation...")
    recall_1, recall_5, recall_10, map_value = _eval_retrieval(Image_X, Image_Y, Text_X, Text_Y)
    text_file.write("Image-to-Text Evaluation \n")
    text_file.write("{:8}{:8.2%}\n".format('Recall@1', recall_1))
    text_file.write("{:8}{:8.2%}\n".format('Recall@5', recall_5))
    text_file.write("{:8}{:8.2%}\n".format('Recall@10', recall_10))
    text_file.write("{:8}{:8.2%}\n".format('MAP', map_value))
    text_file.close()
    """

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Metric learning and evaluate performance")
    parser.add_argument('result_dir',
                        help="Result directory. Containing extracted features and labels. "
                             "CMC curve will also be saved to this directory.")
    parser.add_argument('--method', choices=['euclidean', 'cosine'],
                        default='cosine')
    args = parser.parse_args()
    main(args)
