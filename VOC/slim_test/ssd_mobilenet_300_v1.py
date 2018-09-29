# -- coding: utf-8 --
import numpy as np
import math
from collections import namedtuple
import tensorflow as tf
import ssd_common
import tf_extended as tfe
from nets import custom_layers
slim = tf.contrib.slim

#输入与输出层
IMAGE_SIZE = 300
NUM_CHANNELS = 3
STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

class SSDNet(object):
    """Implementation of the SSD mobilenet-based 224 network.

    The default features layers with 224*224 image input are:
      conv6 ==> 28 x 28
      conv7 ==> 14 x 14
      conv9 ==> 7 x 7
      last ==> 1 x 1
    The default image size used to train this network is 224*224.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=15,
        no_annotation_label=15,
        feat_layers=['part2_new1', 'part2_new3', 'part2_new4', 'part2_new5', 'part2_new6', 'part2_new7'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(10., 21.),
                      (21., 35.),
                      (35., 75.),
                      (75., 115.),
                      (115., 170.),
                      (170., 210.)],
        anchor_ratios=[[2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params
    
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.nn.softmax,
            reuse=None,
            scope='mobile_net_300_v1'):
        """SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)


    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    # if normalization > 0:
    #     net = custom_layers.l2_normalization(net, scaling=True)

    # Number of anchors (default boxes)
    num_anchors = len(sizes) + len(ratios)
    inputs_shape = net.get_shape().as_list()

    # Location offset [x,y,w,h] * boxes_num
    num_loc_pred = num_anchors * 4
    with tf.variable_scope('conv_loc'):
        loc_weights = tf.get_variable('loc_weights', [3, 3, inputs_shape[3], num_loc_pred], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        loc_biases = tf.get_variable('loc_biases', [num_loc_pred], initializer=tf.constant_initializer(0.0))
        loc_conv = tf.nn.conv2d(net, loc_weights, strides=[1,1,1,1], padding='SAME')
        loc_pred = tf.nn.bias_add(loc_conv, loc_biases)
        loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])

    # Class prediction [num_classes] * boxes_num
    num_cls_pred = num_anchors * num_classes
    with tf.variable_scope('conv_cls'):
        cls_weights = tf.get_variable('cls_weights', [3, 3, inputs_shape[3], num_cls_pred], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        cls_biases = tf.get_variable('cls_biases', [num_cls_pred], initializer=tf.constant_initializer(0.1))
        cls_conv = tf.nn.conv2d(net, cls_weights, strides=[1,1,1,1], padding='SAME')
        cls_pred = tf.nn.bias_add(cls_conv, cls_biases)
        cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

    return cls_pred, loc_pred

def separable_conv2d(inputs, dw_size, pw_size, downsample=False, is_training=True, padding='SAME',scope=''):
    _stride = [1,2,2,1] if downsample else [1,1,1,1]
    with tf.variable_scope(scope):
        dw_filter = tf.get_variable("dw", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        dw_net = tf.nn.depthwise_conv2d(inputs, dw_filter, _stride, padding=padding, name='dw_net', data_format='NHWC')
        dw_bn = tf.contrib.layers.batch_norm(dw_net, decay=BN_DECAY, center=True, scale=True, is_training=is_training, scope='dw_bn')
        dw_active = ACTIVATION(dw_bn)

        pw_filter = tf.get_variable("pw", pw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        pw_net = tf.nn.conv2d(dw_active, pw_filter, strides=[1,1,1,1], padding='SAME')
        pw_bn = tf.contrib.layers.batch_norm(pw_net, decay=BN_DECAY, center=True, scale=True, is_training=is_training, scope='pw_bn')
        pw_active = ACTIVATION(pw_bn)
    return pw_active, dw_filter, pw_filter

def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.nn.softmax,
            reuse=None,
            scope='mobile_net_300_v1'):
    """SSD net definition.
    """
    end_points = {}
    restore_variables = {}
    # Net Structure
    with tf.variable_scope(scope):
        '''
            1. Full Conv
        '''
        with tf.variable_scope('part_1'):    
            with tf.variable_scope('conv_1'):
                conv1_weights = tf.get_variable("weight", [3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
                conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
                net = tf.nn.conv2d(inputs, conv1_weights, strides=[1,1,1,1], padding='SAME')
                net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
                end_points['part1_conv1'] = net
                restore_variables['part1_conv1_weights'] = conv1_weights
                restore_variables['part1_conv1_bias'] = conv1_biases
            with tf.variable_scope('conv_2'):
                conv2_weights = tf.get_variable("weight", [3, 3, 32, 32], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
                conv2_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
                net = tf.nn.conv2d(net, conv2_weights, strides=[1,2,2,1], padding='SAME')
                net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))
                end_points['part1_conv2'] = net
                restore_variables['part1_conv2_weights'] = conv2_weights
                restore_variables['part1_conv2_bias'] = conv2_biases
        '''
            2. Separable Conv
        '''
        with tf.variable_scope('part_2'):
            net, dw_1, pw_1 = separable_conv2d(net, [3, 3, 32, 1], [1, 1, 32, 64], is_training=is_training, scope = 'conv_1')
            end_points['part2_conv1'] = net
            net, dw_2, pw_2 = separable_conv2d(net, [3, 3, 64, 1], [1, 1, 64, 128], downsample=True, is_training=is_training, scope='conv_2')
            end_points['part2_conv2'] = net
            net, dw_3, pw_3 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 128], is_training=is_training, scope='conv_3')
            end_points['part2_conv3'] = net
            net, dw_4, pw_4 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 256], downsample=True, is_training=is_training, scope='conv_4')
            end_points['part2_conv4'] = net

            net, dw_5_1, pw_5_1 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_5_1')
            end_points['part2_conv51'] = net
            net, dw_5_2, pw_5_2 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_5_2')
            end_points['part2_conv52'] = net
            net, dw_5_3, pw_5_3 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_5_3')
            end_points['part2_conv53'] = net
            net, dw_5_4, pw_5_4 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_5_4')
            end_points['part2_conv54'] = net
            net, dw_5_5, pw_5_5 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_5_5')  
            end_points['part2_conv55'] = net

            net, dw_6, pw_6 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 512], is_training=is_training, scope='new_1')   #38*38*512
            end_points['part2_new1'] = net
            net, dw_7, pw_7 = separable_conv2d(net, [3, 3, 512, 1], [1, 1, 512, 1024], downsample=True, is_training=is_training, scope='new_2')
            end_points['part2_new2'] = net
            net, dw_8, pw_8 = separable_conv2d(net, [3, 3, 1024, 1], [1, 1, 1024, 1024], is_training=is_training, scope='new_3') #19*19*1024
            end_points['part2_new3'] = net
            net, dw_9, pw_9 = separable_conv2d(net, [3, 3, 1024, 1], [1, 1, 1024, 512], downsample=True, is_training=is_training, scope='new_4') #10*10*512
            end_points['part2_new4'] = net
            net, dw_10, pw_10 = separable_conv2d(net, [3, 3, 512, 1], [1, 1, 512, 256], downsample=True, is_training=is_training, scope='new_5') #5*5*256
            end_points['part2_new5'] = net
            net, dw_11, pw_11 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], downsample=True, is_training=is_training, scope='new_6') #3*3*256
            end_points['part2_new6'] = net
            net, dw_12, pw_12 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], downsample=True, is_training=is_training, padding='VALID', scope='new_7') #1*1*256
            end_points['part2_new7'] = net
            # need restore
            restore_variables['part2_conv1_dw'] = dw_1
            restore_variables['part2_conv1_pw'] = pw_1
            restore_variables['part2_conv2_dw'] = dw_2
            restore_variables['part2_conv2_pw'] = pw_2
            restore_variables['part2_conv3_dw'] = dw_3
            restore_variables['part2_conv3_pw'] = pw_3
            restore_variables['part2_conv4_dw'] = dw_4
            restore_variables['part2_conv4_pw'] = pw_4
            restore_variables['part2_conv51_dw'] = dw_5_1
            restore_variables['part2_conv51_pw'] = pw_5_1
            restore_variables['part2_conv52_dw'] = dw_5_2
            restore_variables['part2_conv52_pw'] = pw_5_2
            restore_variables['part2_conv53_dw'] = dw_5_3
            restore_variables['part2_conv53_pw'] = pw_5_3
            restore_variables['part2_conv54_dw'] = dw_5_4
            restore_variables['part2_conv54_pw'] = pw_5_4
            restore_variables['part2_conv55_dw'] = dw_5_5
            restore_variables['part2_conv55_pw'] = pw_5_5

            restore_variables['part2_new1_dw'] = dw_6
            restore_variables['part2_new1_pw'] = pw_6
            restore_variables['part2_new2_dw'] = dw_7
            restore_variables['part2_new2_pw'] = pw_7
            restore_variables['part2_new3_dw'] = dw_8
            restore_variables['part2_new3_pw'] = pw_8
            restore_variables['part2_new4_dw'] = dw_9
            restore_variables['part2_new4_pw'] = pw_9
            restore_variables['part2_new5_dw'] = dw_10
            restore_variables['part2_new5_pw'] = pw_10
            restore_variables['part2_new6_dw'] = dw_11
            restore_variables['part2_new6_pw'] = pw_11
            restore_variables['part2_new7_dw'] = dw_12
            restore_variables['part2_new7_pw'] = pw_12

    # Prediction and localisations layers.
    predictions = []
    logits = []
    localisations = []
    for i, layer in enumerate(feat_layers):
        with tf.variable_scope(layer + '_box'):
            p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))    # add softmax()
            logits.append(p)
            localisations.append(l)


    return predictions, localisations, logits, end_points, restore_variables

def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids; 
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # anchor center point(ratio 0~1)
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)


    return y, x, h, w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        print lshape
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)




