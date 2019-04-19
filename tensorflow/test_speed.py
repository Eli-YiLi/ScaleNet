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
import numpy as np
import os
import time
from resnet_v1 import resnet_v1_50,resnet_v1_101
from seresnet import SE_ResNet
from resnext import SE_ResNeXt
from  scale_resnet_v1 import scale_resnet_v1_50
import sys

tf.reset_default_graph()

model = sys.argv[1]
input_image = tf.placeholder(tf.float32,  shape = (None, 224, 224, 3), name = 'input_placeholder')
if model == 'se':
    outputs = SE_ResNet(input_image, 1000, is_training = False, data_format='channels_last')
elif model == 'res':
    outputs = resnet_v1_50(input_image, 1000, is_training = False, scope='resnet_v1_50')
elif model == 'scale':
    outputs = scale_resnet_v1_50(input_image, 1000, is_training = False, scope='resnet_v1_50')
elif model == 'next':
    outputs = SE_ResNeXt(input_image, 1000, is_training = False, data_format='channels_last')

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    img_dict = {input_image: np.random.randn(16, 224, 224, 3)}
    t1 = time.time()
    for i in range(1000):
        predict = sess.run(outputs, feed_dict = img_dict)
    t2 = time.time()
    print((t2-t1)/1000)
