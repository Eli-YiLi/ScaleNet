# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import scale_resnet_utils


scale_resnet_arg_scope = scale_resnet_utils.scale_resnet_arg_scope
slim = tf.contrib.slim

channel_num = {
'1_1': [62,9,5,12,56],
'1_2': [55,27,5,1,56],
'1_3': [59,26,0,3,28],
'2_1': [125,41,6,3,28],
'2_2': [90,39,9,37,28],
'2_3': [106,56,4,9,28],
'2_4': [116,56,3,0,14],
'3_1': [223,71,55,0,14],
'3_2': [196,104,44,5,14],
'3_3': [195,98,52,4,14],
'3_4': [155,128,66,0,14],
'3_5': [134,129,86,0,14],
'3_6': [120,127,98,4,7],
'4_1': [237,354,106,0,7],
'4_2': [172,435,90,0,7],
'4_3': [138,462,97,0,7],
}

#channel_num = {
#'1_1': [64,0,0,0,56],
#'1_2': [64,0,0,0,56],
#'1_3': [64,0,0,0,28],
#'2_1': [128,0,0,0,28],
#'2_2': [128,0,0,0,28],
#'2_3': [128,0,0,0,28],
#'2_4': [128,0,0,0,14],
#'3_1': [256,0,0,0,14],
#'3_2': [256,0,0,0,14],
#'3_3': [256,0,0,0,14],
#'3_4': [256,0,0,0,14],
#'3_5': [256,0,0,0,14],
#'3_6': [256,0,0,0,7],
#'4_1': [512,0,0,0,7],
#'4_2': [512,0,0,0,7],
#'4_3': [512,0,0,0,7],
#}

#channel_num = {
#'1_1': [61,11,7,7,56],
#'1_2': [56,23,4,3,56],
#'1_3': [59,24,3,0,28],
#'2_1': [123,41,1,6,28],
#'2_2': [126,38,1,6,28],
#'2_3': [127,41,3,0,28],
#'2_4': [127,41,3,0,14],
#'3_1': [220,86,35,0,14],
#'3_2': [186,64,55,36,14],
#'3_3': [156,25,53,107,14],
#'3_4': [191,44,52,54,14],
#'3_5': [181,53,83,24,14],
#'3_6': [221,82,34,4,14],
#'3_7': [177,62,90,12,14],
#'3_8': [130,75,102,34,14],
#'3_9': [206,71,55,9,14],
#'3_10': [203,83,53,2,14],
#'3_11': [207,73,54,7,14],
#'3_12': [245,84,12,0,14],
#'3_13': [221,103,17,0,14],
#'3_14': [221,100,20,0,14],
#'3_15': [158,99,84,0,14],
#'3_16': [220,106,15,0,14],
#'3_17': [173,92,73,3,14],
#'3_18': [135,122,84,0,14],
#'3_19': [109,71,132,29,14],
#'3_20': [147,94,93,7,14],
#'3_21': [191,108,42,0,14],
#'3_22': [127,95,113,6,14],
#'3_23': [203,117,21,0,7],
#'4_1': [282,377,23,0,7],
#'4_2': [279,388,15,0,7],
#'4_3': [84,442,155,1,7],
#}

class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               block_id,
               unit_id,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = scale_resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    sub = []
    if stride != 1:
        residual = slim.max_pool2d(residual, [stride, stride], padding='VALID', stride=stride, scope='conv2_pool')
    if channel_num['{}_{}'.format(block_id, unit_id)][0] != 0:
        sub.append(scale_resnet_utils.conv2d_same_wo_bnrelu(residual, channel_num['{}_{}'.format(block_id, unit_id)][0], 3, 1,
                                            rate=rate, scope='conv2_1'))
    if channel_num['{}_{}'.format(block_id, unit_id)][1] != 0:
        down = slim.max_pool2d(residual, [2, 2], stride=2, padding='VALID', scope='conv2_2_down')
        mid = scale_resnet_utils.conv2d_same_wo_bnrelu(down, channel_num['{}_{}'.format(block_id, unit_id)][1], 3, 1,
                                            rate=rate, scope='conv2_2')
        feature_size = channel_num['{}_{}'.format(block_id, unit_id)][4]
        with tf.variable_scope('conv2_2_up'):
            up = tf.image.resize_images(mid, [feature_size, feature_size])
        sub.append(up)
    if channel_num['{}_{}'.format(block_id, unit_id)][2] != 0:
        down = slim.max_pool2d(residual, [4, 4], stride=4, padding='VALID', scope='conv2_3_down')
        mid = scale_resnet_utils.conv2d_same_wo_bnrelu(down, channel_num['{}_{}'.format(block_id, unit_id)][2], 3, 1,
                                            rate=rate, scope='conv2_3')
        feature_size = channel_num['{}_{}'.format(block_id, unit_id)][4]
        with tf.variable_scope('conv2_3_up'):
            up = tf.image.resize_images(mid, [feature_size, feature_size])
        sub.append(up)
    if channel_num['{}_{}'.format(block_id, unit_id)][3] != 0:
        down = slim.max_pool2d(residual, [7, 7], stride=7, padding='VALID', scope='conv2_4_down')
        mid = scale_resnet_utils.conv2d_same_wo_bnrelu(down, channel_num['{}_{}'.format(block_id, unit_id)][3], 3, 1,
                                            rate=rate, scope='conv2_4')
        feature_size = channel_num['{}_{}'.format(block_id, unit_id)][4]
        with tf.variable_scope('conv2_4_up'):
            up = tf.image.resize_images(mid, [feature_size, feature_size])
        sub.append(up)

    residual = tf.concat(sub, axis=-1)
    #residual = scale_resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
    #                                    rate=rate, scope='conv2')
    #residual = tf.Print(residual, [tf.shape(residual)], message='conv2 shape: ', summarize=1)
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def scale_resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode. If this is set
      to None, the callers can specify slim.batch_norm's is_training parameter
      from an outer slim.arg_scope.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'scale_resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         scale_resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
            if is_training is not None else NoOpScope()):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = scale_resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
        net = scale_resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                                              store_non_strided_activations)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points
scale_resnet_v1.default_image_size = 224


def scale_resnet_v1_block(scope, base_depth, num_units, stride, block_id):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return scale_resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'block_id': block_id,
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'block_id': block_id,
  }])


def scale_resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='scale_resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      scale_resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, block_id=1),
      scale_resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, block_id=2),
      scale_resnet_v1_block('block3', base_depth=256, num_units=6, stride=2, block_id=3),
      scale_resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, block_id=4),
  ]
  return scale_resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
scale_resnet_v1_50.default_image_size = scale_resnet_v1.default_image_size

#def scale_resnet_v1_101(inputs,
#                 num_classes=None,
#                 is_training=True,
#                 global_pool=True,
#                 output_stride=None,
#                 spatial_squeeze=True,
#                 store_non_strided_activations=False,
#                 reuse=None,
#                 scope='scale_resnet_v1_101'):
#  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
#  blocks = [
#      scale_resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, block_id=1),
#      scale_resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, block_id=2),
#      scale_resnet_v1_block('block3', base_depth=256, num_units=23, stride=2, block_id=3),
#      scale_resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, block_id=4),
#  ]
#  return scale_resnet_v1(inputs, blocks, num_classes, is_training,
#                   global_pool=global_pool, output_stride=output_stride,
#                   include_root_block=True, spatial_squeeze=spatial_squeeze,
#                   store_non_strided_activations=store_non_strided_activations,
#                   reuse=reuse, scope=scope)
#scale_resnet_v1_101.default_image_size = scale_resnet_v1.default_image_size
