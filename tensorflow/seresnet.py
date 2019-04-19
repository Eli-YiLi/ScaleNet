# Copyright 2018 Changan Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

USE_FUSED_BN = True
BN_EPSILON = 9.999999747378752e-06
BN_MOMENTUM = 0.99

# input image order: BGR, range [0-255]
# mean_value: 104, 117, 123
# only subtract mean is used
# for root block, use dummy input_filters, e.g. 128 rather than 64 for the first block
def se_bottleneck_block(inputs, input_filters, name_prefix, is_training, data_format='channels_last', need_reduce=True, is_root=False, reduced_scale=16):
    bn_axis = -1 if data_format == 'channels_last' else 1
    strides_to_use = 1
    residuals = inputs
    if need_reduce:
        strides_to_use = 1 if is_root else 2
        proj_mapping = tf.layers.conv2d(inputs, input_filters * 2, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        residuals = tf.layers.batch_normalization(proj_mapping, momentum=BN_MOMENTUM,
                                name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    reduced_inputs = tf.layers.conv2d(inputs, input_filters / 2, (1, 1), use_bias=False,
                            name=name_prefix + '_1x1_reduce', strides=(strides_to_use, strides_to_use),
                            padding='valid', data_format=data_format, activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer())
    reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=BN_MOMENTUM,
                                        name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                        epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')


    conv3_inputs = tf.layers.conv2d(reduced_inputs_relu, input_filters / 2, (3, 3), use_bias=False,
                                name=name_prefix + '_3x3', strides=(1, 1),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=BN_MOMENTUM, name=name_prefix + '_3x3/bn',
                                        axis=bn_axis, epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


    increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_increase', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=BN_MOMENTUM,
                                        name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                        epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    if data_format == 'channels_first':
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
    else:
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

    down_inputs = tf.layers.conv2d(pooled_inputs, (input_filters * 2) // reduced_scale, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_down', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


    up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_up', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

    rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
    pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
    return tf.nn.relu(pre_act, name=name_prefix + '/relu')
    #return tf.nn.relu(residuals + prob_outputs * increase_inputs_bn, name=name_prefix + '/relu')

def SE_ResNet(input_image, num_classes, is_training=False, data_format='channels_last', net_depth=50):
    bn_axis = -1 if data_format == 'channels_last' else 1

    # the input image should in BGR order, note that this is not the common case in Tensorflow
    # convert from RGB to BGR
    if data_format == 'channels_last':
        image_channels = tf.unstack(input_image, axis=-1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1)
    else:
        image_channels = tf.unstack(input_image, axis=1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)

    if net_depth not in [50, 101]:
        raise TypeError('Only ResNet50 or ResNet101 is supprted now.')
    input_depth = [128, 256, 512, 1024] # the input depth of the the first block is dummy input
    num_units = [3, 4, 6, 3] if net_depth==50 else [3, 4, 23, 3]
    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']

    if data_format == 'channels_first':
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]])
    else:
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])

    inputs_features = tf.layers.conv2d(swaped_input_image, input_depth[0]//2, (7, 7), use_bias=False,
                                name='conv1/7x7_s2', strides=(2, 2),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())

    inputs_features = tf.layers.batch_normalization(inputs_features, momentum=BN_MOMENTUM,
                                        name='conv1/7x7_s2/bn', axis=bn_axis,
                                        epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs_features = tf.nn.relu(inputs_features, name='conv1/relu_7x7_s2')

    inputs_features = tf.layers.max_pooling2d(inputs_features, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool1/3x3_s2')

    is_root = True
    for ind, num_unit in enumerate(num_units):
        need_reduce = True
        for unit_index in range(1, num_unit+1):
            inputs_features = se_bottleneck_block(inputs_features, input_depth[ind], block_name_prefix[ind].format(unit_index), is_training=is_training, data_format=data_format, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False

    if data_format == 'channels_first':
        pooled_inputs = tf.reduce_mean(inputs_features, [2, 3], name='pool5/7x7_s1', keep_dims=True)
    else:
        pooled_inputs = tf.reduce_mean(inputs_features, [1, 2], name='pool5/7x7_s1', keep_dims=True)

    pooled_inputs = tf.contrib.layers.flatten(pooled_inputs)

    logits_output = tf.layers.dense(pooled_inputs, num_classes,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True)

    return logits_output, tf.nn.softmax(logits_output, name='prob')

'''run test for the chcekpoint again
'''
#import numpy as np
#import os
#import time
#
#tf.reset_default_graph()
#
#input_image = tf.placeholder(tf.float32,  shape = (None, 3, 224, 224), name = 'input_placeholder')
#outputs = SE_ResNet(input_image, 1000, is_training = False, data_format='channels_first')
#
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    img_dict = {input_image: np.random.randn(16,3,224,224)}
#    t1 = time.time()
#    for i in range(100):
#        predict = sess.run(outputs, feed_dict = img_dict)
#    t2 = time.time()
#    print((t2-t1)/100)
