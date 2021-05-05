"""
    Network Modules of CNN, Bi-LSTM, and Loss Functions
    Some code about LSTM comes from https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nets import nets_factory
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

import pdb 

def pairwise_distance(A, B):
    """ Pairwise distance between A and B
    """
    rA = tf.reduce_sum(A * A, 1, keepdims=True)

    rB = tf.reduce_sum(B * B, 1, keepdims=True)

    # turn r into column vector
    D = rA - 2 * tf.matmul(A, tf.transpose(B)) + tf.transpose(rB)

    return D


def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.
    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.
    Args:
      state_size: TensorShape or int that specifies the size of a tensor.
      prefix: optional additional list of dimensions to prepend.
    Returns:
      result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size


def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer


def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
            for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                           flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state


def bidirectional_lstm(input, hidden_state_dimension, sequence_length, reuse=False):
    """
        Bi-directional LSTM
        Referred https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    state_initializer = make_variable_state_initializer()
    with tf.variable_scope("bidirectional_lstm", reuse=reuse):
        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_state_dimension,
                                                                    state_is_tuple=True)
                if FLAGS.is_training:
                    lstm_cell[direction] = tf.contrib.rnn.DropoutWrapper(lstm_cell[direction],
                                                                         input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                                                                         output_keep_prob=FLAGS.lstm_dropout_keep_prob)

                initial_state[direction] = get_initial_cell_state(lstm_cell[direction], state_initializer,
                                                                  FLAGS.batch_size, tf.float32)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        (outputs_forward, outputs_backward), (final_states_forward, final_states_backward) = \
            tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                            lstm_cell["backward"],
                                            input,
                                            dtype=tf.float32,
                                            sequence_length=sequence_length,
                                            initial_state_fw=initial_state["forward"],
                                            initial_state_bw=initial_state["backward"])
        # batch_size * T * 1024
        output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        states = tf.reduce_max(output, axis=1, name='mean_states')

    return states


def generate_fixed_filter():

    fixed_filter = tf.convert_to_tensor(np.array([1.0/FLAGS.group_size]),dtype=tf.float32)
    fixed_filter = tf.reshape(tf.tile(fixed_filter,[FLAGS.batch_size*FLAGS.group_size]),(FLAGS.batch_size,FLAGS.group_size))
    return fixed_filter


def generate_uniform_random_filter():
    random_filter = tf.random_uniform((FLAGS.batch_size,FLAGS.group_size),minval=0,maxval=1,dtype=tf.float32)
    return random_filter


def generate_normal_random_filter():
    random_filter = tf.truncated_normal((FLAGS.batch_size,FLAGS.group_size),mean=0.5,stddev=0.25)
    return random_filter


def build_text_features(input_seqs, input_mask):
    """ Builds the Bi-directional LSTM and extract text features

    Args:
        input_seqs: input sequences of text ids
        input_mask: mask to distinguish real words from padding words.

    Returns:
        features: extracted text features
        end_points: end points
    """
    end_points = {}
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("BiLSTM"):
        token_embedding_weights = tf.get_variable(name="token_embedding_weights",
                                                  shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                                                  initializer=initializer)
        end_points['token_embedding_weights'] = token_embedding_weights

        token_lstm_input = tf.nn.embedding_lookup(token_embedding_weights, input_seqs)
        end_points['token_lstm_input'] = token_lstm_input

        sequence_length = tf.reduce_sum(input_mask, 1)
        states = bidirectional_lstm(token_lstm_input,
                                    hidden_state_dimension=FLAGS.num_lstm_units,
                                    sequence_length=sequence_length)

        features = tf.expand_dims(tf.expand_dims(states, 1), 1)  # batch_size * 1 x 1 x 1024

        return features, end_points


def build_image_features(network_fn, images):
    """Builds the CNN model and extract image features

    Args:
        network_fn : CNN network
        images: images
    Returns:
        net: extracted image features
        end_points: end points
    """
    # Extract Image features from CNN networks
    features, end_points = network_fn(images, scope=FLAGS.model_scope)  # batch_size * 1 x 1 x 1024
    image_features_3d = end_points['FeatureMap']
    return features, end_points, image_features_3d


def build_joint_embeddings(inputs, scope=None):
    """Build joint embeddings

    Args:
        inputs: input features   batch_size * 1 * 1 * 1024
        scope: name of the scope

    Returns:
        joint_embeddings: reduced features for joint embedding learning
    """
    with tf.variable_scope("joint_embedding"):
        arg_scope = nets_factory.arg_scopes_map[FLAGS.model_name](weight_decay=FLAGS.weight_decay)
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.batch_norm], is_training=FLAGS.is_training):
                if scope=='image_joint_embedding':
                    joint_embeddings = slim.conv2d(inputs, FLAGS.feature_size, [1, 1],
                                               activation_fn=None, scope=scope)
                    joint_embeddings = tf.layers.batch_normalization(joint_embeddings,training=FLAGS.is_training, name='image_bn')
                    joint_embeddings = tf.nn.tanh(joint_embeddings, name='image_tanh')
                if scope=='text_joint_embedding':
                    joint_embeddings = slim.conv2d(inputs, FLAGS.feature_size, [1, 1], 
                                               activation_fn=None, scope=scope)

                    joint_embeddings = tf.layers.batch_normalization(joint_embeddings,training=FLAGS.is_training, name='text_bn')
                    joint_embeddings = tf.nn.tanh(joint_embeddings, name='text_tanh')
                joint_embeddings = tf.squeeze(joint_embeddings, [1, 2])  # batch_size * feature_size

    return joint_embeddings


def build_part_conv_features(image_features_3d):
    """Build part conv features and image part embeddings
    Args:
        image_features_3d: network 3d output
    Returns:
        fc_layers: part embeddings
        conv_layers: part conv features
    """
    with tf.variable_scope("build_conv_features"):

        fc_layers = []
        
        # print(image_features_3d.shape)
        interval = int(FLAGS.image_height/32/FLAGS.group_size)
        print("interval ", interval)
        for i in range(FLAGS.group_size):
            image_part_feature = image_features_3d[:,i*interval:(i*interval+interval)]
            
            # FC1
            conv_layer = slim.conv2d(image_part_feature, FLAGS.feature_size, [1, 1], activation_fn=None, scope='part_conv_layer%d'%i)
            conv_layer = tf.layers.batch_normalization(conv_layer,training=FLAGS.is_training)
            conv_layer = tf.nn.relu6(conv_layer)   # what's the difference between relu and relu6
            # conv 1*1 to reduce dims
            conv_layer = tf.reduce_mean(conv_layer, [1, 2], keepdims=True, name='part_average_pooling')

            # FC2
            fc_layer = slim.conv2d(conv_layer, FLAGS.feature_size, [1, 1],
                                               activation_fn=None, scope='part_fc_layer%d'%i)
            fc_layer = tf.layers.batch_normalization(fc_layer,training=FLAGS.is_training)
            fc_layer = tf.nn.tanh(fc_layer)
            fc_layer = tf.squeeze(fc_layer,[1,2])
            fc_layers.append(fc_layer)
            
        fc_layers = tf.convert_to_tensor(fc_layers)
    
        return fc_layers


def build_text_coefficient_for_part(text_features, group_size):
    with tf.variable_scope("build_text_coefficient_for_part"):
        
        coef_conv1 = slim.conv2d(text_features, FLAGS.feature_size * 2, [1, 1],                   
                                               activation_fn=None,scope='coef_conv1')   # using 1 dim con2d to substitude the fully connected layer4
        # coft_conv1's dim is [batch_size,1,1,256]  FLAGS.feature_size=512
        coef_conv1 = tf.layers.batch_normalization(coef_conv1,training=FLAGS.is_training)    
        coef_conv1 = tf.nn.relu(coef_conv1)
        coef_conv2 = slim.conv2d(coef_conv1, group_size, [1, 1],
                                               activation_fn=tf.nn.sigmoid,scope='coef_conv2')
            
        coefficient = tf.squeeze(coef_conv2, [1,2])
       
        return coefficient


def build_image_part_dc(fc_layers_part, text_coefficient_part):
    with tf.variable_scope("build_image_part_dc"):
        # build image dynamic convolution

        output = tf.einsum('bg,gbf->bgf', text_coefficient_part, fc_layers_part) # group_size * batch_size * feature_size
        image_part_embeddings_for_loss = output
        image_part_embeddings = tf.reduce_sum(output,axis=1)  
        
        # because keepdim=false and the element num in first dim is 1 ,the output's dim is batch_size*feature_size
        return image_part_embeddings,image_part_embeddings_for_loss

def cmpm_loss_compute(text_embeddings, image_embeddings, labels):
    """ Cross-Modal Projection Matching Loss (CMPM)

    Args:
        text_embeddings: text joint embeddings
        image_embeddings: image joint embeddings
        labels: class labels

    Returns:
        i2t_matching_loss: cmpm loss for image projected to text
        t2i_matching_loss: cmpm loss for text projected to image
        pos_avg_dist: average distance of positive pairs
        neg_avg_dist: average distance of negative pairs
    """
    # label mask
    batch_size = image_embeddings.get_shape().as_list()[0]
    mylabels = tf.cast(tf.reshape(labels, [batch_size, 1]), tf.float32)
    labelD = pairwise_distance(mylabels, mylabels)
    label_mask = tf.cast(tf.less(labelD, 0.5), tf.float32)  # 1-match   0-unmatch

    # cross-modal scalar projection
    image_embeddings_norm = tf.nn.l2_normalize(image_embeddings, dim=-1)
    text_embeddings_norm = tf.nn.l2_normalize(text_embeddings, dim=-1)

    image_proj_text = tf.matmul(image_embeddings, tf.transpose(text_embeddings_norm))
    text_proj_image = tf.matmul(text_embeddings, tf.transpose(image_embeddings_norm))

    # softmax, higher scalar projection gives higher probability
    i2t_pred = tf.nn.softmax(image_proj_text)
    t2i_pred = tf.nn.softmax(text_proj_image)

    # normalize the true matching distribution
    label_mask = tf.divide(label_mask, tf.reduce_sum(label_mask, axis=1, keepdims=True))

    # KL Divergence
    i2t_matching_loss = tf.reduce_mean(tf.reduce_sum(i2t_pred * tf.log(1e-8 + i2t_pred / (label_mask + 1e-8)), 1))
    t2i_matching_loss = tf.reduce_mean(tf.reduce_sum(t2i_pred * tf.log(1e-8 + t2i_pred / (label_mask + 1e-8)), 1))

    # averaged cosine distance of positive and negative pairs for observation
    cosdist = 1.0 - tf.matmul(text_embeddings_norm, tf.transpose(image_embeddings_norm))

    pos_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.less(labelD, 0.5)))
    neg_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.greater(labelD, 0.5)))

    return i2t_matching_loss, t2i_matching_loss, pos_avg_dist, neg_avg_dist


def cmpc_loss_compute(text_embeddings, image_embeddings, labels):
    """ Cross-Modal Projection Classification loss (CMPC)

    Args:
        text_embeddings: text joint embeddings
        image_embeddings: image joint embeddings
        labels: class labels

    Returns:
        ipt_loss: cmpc loss for image projected to text
        tpi_loss: cmpc loss for text projected to image
        image_precision: precision of image classification
        text_precision: precision of text classification
    """
    # norm-softmax
    feature_size = image_embeddings.get_shape().as_list()[-1]
    W = tf.get_variable("Wfc", shape=[feature_size, FLAGS.num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
    W_norm = tf.nn.l2_normalize(W, dim=0)

    # cross-modal vector projection
    image_embeddings_norm = tf.nn.l2_normalize(image_embeddings, dim=-1)
    text_embeddings_norm = tf.nn.l2_normalize(text_embeddings, dim=-1)

    image_proj_text = tf.multiply(tf.reduce_sum(tf.multiply(image_embeddings, text_embeddings_norm),
                                                axis=1, keepdims=True), text_embeddings_norm)
    text_proj_image = tf.multiply(tf.reduce_sum(tf.multiply(text_embeddings, image_embeddings_norm),
                                                axis=1, keepdims=True), image_embeddings_norm)

    # classification Loss
    image_logits = tf.matmul(image_proj_text, W_norm)
    text_logits = tf.matmul(text_proj_image, W_norm)

    one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes)

    ipt_loss = tf.losses.softmax_cross_entropy(
                logits=image_logits, onehot_labels=one_hot_labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)

    tpi_loss = tf.losses.softmax_cross_entropy(
                logits=text_logits, onehot_labels=one_hot_labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)

    # classification accuracy for observation
    image_pred = slim.softmax(image_logits, scope='image_predictions')
    text_pred = slim.softmax(text_logits, scope='text_predictions')

    image_predictions = tf.argmax(image_pred, axis=1)
    image_precision = tf.reduce_mean(tf.to_float(tf.equal(image_predictions, labels)))

    text_predictions = tf.argmax(text_pred, axis=1)
    text_precision = tf.reduce_mean(tf.to_float(tf.equal(text_predictions, labels)))

    return ipt_loss, tpi_loss, image_precision, text_precision

def image_ID_loss_compute(image_features, labels):
    with tf.variable_scope("image_ID_loss"):
        image_features = tf.squeeze(image_features,[1,2])
        one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes)

        W = tf.get_variable("W_image", shape=[FLAGS.feature_size*2, FLAGS.num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
        W_norm = tf.nn.l2_normalize(W, dim=0)
        image_logits = tf.matmul(image_features, W_norm)
        image_logits = tf.convert_to_tensor(image_logits,tf.float32)
        image_ID_loss = tf.losses.softmax_cross_entropy(
                logits=image_logits, onehot_labels=one_hot_labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
        return image_ID_loss

def text_ID_loss_compute(text_features, labels):
    with tf.variable_scope("text_ID_loss"):
        text_features = tf.squeeze(text_features,[1,2])
        one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes)

        W = tf.get_variable("W_text", shape=[FLAGS.feature_size*2, FLAGS.num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
        W_norm = tf.nn.l2_normalize(W, dim=0)
        text_logits = tf.matmul(text_features, W_norm)
        text_logits = tf.convert_to_tensor(text_logits,tf.float32)
        text_ID_loss = tf.losses.softmax_cross_entropy(
                logits=text_logits, onehot_labels=one_hot_labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
        return text_ID_loss

def fc_similarity_loss(fc_layers):
    fc_layers = tf.nn.l2_normalize(fc_layers,dim=-1)
    fc_layers = tf.einsum('gbf->bgf',fc_layers)
    fc_layers_transpose = tf.einsum('bgf->bfg',fc_layers)
    matmul_result = tf.einsum('bgf,bfm->bgm',fc_layers,fc_layers_transpose)
    identity_mat = tf.eye(FLAGS.group_size,FLAGS.group_size,batch_shape=[FLAGS.batch_size])
    #fc_loss = tf.reduce_mean(tf.square(tf.subtract(matmul_result,identity_mat)))
    fc_loss = tf.losses.mean_squared_error(matmul_result,identity_mat)
    return fc_loss

def build_image_part_cls(image_conv_layers):
    with tf.variable_scope("build_image_part_cls"):
        image_part_cls_logits = []
        W = tf.get_variable("W_part", shape=[FLAGS.feature_size, FLAGS.num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        W_norm = tf.nn.l2_normalize(W, dim=0)
        for i in range(FLAGS.group_size):
            image_dc_embeddings = image_conv_layers[i]
            image_part_cls_logit = tf.matmul(image_dc_embeddings, W_norm)
            image_part_cls_logits.append(image_part_cls_logit)
        image_part_cls_logits = tf.convert_to_tensor(image_part_cls_logits,tf.float32)

        return image_part_cls_logits

def image_part_loss_compute(image_part_logits, labels):
    """ image part loss

    Args:
        image_part_logits_set: set of image part logits
        labels: person id labels

    Returns:
        image_part_loss: part loss for image classifier
    """
    image_part_cls_loss = 0.0
    
    labels = tf.tile(tf.expand_dims(labels,0),[FLAGS.group_size,1])

    # labels = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(labels,-1),[1,FLAGS.group_size]), (FLAGS.batch_size*FLAGS.group_size,1)))
    one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    image_part_cls_loss = tf.losses.softmax_cross_entropy(
                logits=image_part_logits, onehot_labels=one_hot_labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
    # image_part_loss = tf.reduce_mean(image_part_cls_loss)
    image_part_pred = slim.softmax(image_part_logits, scope='image_part_predictions')
    image_part_predictions = tf.argmax(image_part_pred, axis=2)
    image_part_precision = tf.reduce_mean(tf.to_float(tf.equal(image_part_predictions, labels)))

    return image_part_cls_loss, image_part_precision

