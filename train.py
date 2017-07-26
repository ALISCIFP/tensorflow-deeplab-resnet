"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
import os
import re
import time

import numpy as np
import tensorflow as tf

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)  # VOC2012
# IMG_MEAN = np.array((40.9729668,   42.62135134,  40.93294311), dtype=np.float32) #ILD
#IMG_MEAN = np.array((88.89328702, 89.36887475, 88.8973059), dtype=np.float32)  # LUNA16
# IMG_MEAN = np.array((109.5388, 118.6897, 124.6901), dtype=np.float32)  # ImageNet2016 Scene-parsing Mean
#
# LUNA16_softmax_weights = np.array((2.15129033634559E-05, 0.0002845522, 0.0002506645, 0.0123730652, 0.9870702051),
#                                   dtype=np.float32)
#IMG_MEAN = np.array((109.5388, 118.6897, 124.6901), dtype=np.float32)  # ImageNet2016 Scene-parsing Mean
IMG_MEAN = np.array((70.49377469, 70.51345116,  70.66025172), dtype=np.float32) #LITS

#[ 69.9417258   70.08041571  69.92282781] #LITS PNG format


#LUNA16_softmax_weights = np.array((2.15129033634559E-05, 0.0002845522, 0.0002506645, 0.0123730652, 0.9870702051),dtype=np.float32)
LUNA16_softmax_weights = np.ones(3,dtype=np.float32)
#LUNA16_softmax_weights = np.array((0.00120125,  0.02164801,0.97715074),dtype=np.float32) #[15020370189   332764489    18465194]
#LUNA16_softmax_weights = np.array((0.00116335,  0.05251166,  0.946325),dtype=np.float32) #[15020370189   332764489    18465194]


GPU_MASK = '1'
BATCH_SIZE = 4
DATA_DIRECTORY = None
DATA_LIST_PATH = None
VAL_DATA_LIST_PATH = None
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 1000000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 100
VAL_INTERVAL = 11
SNAPSHOT_DIR = None
WEIGHT_DECAY = 0.0005


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    # imPred = imPred * (imLab > 0)

    # Compute area intersection:
    # print(np.unique(imPred))
    # print(np.unique(imLab))

    intersection = np.copy(imPred)
    intersection[imPred != imLab] = -1
    # print(np.unique(intersection))
    # print("--------------------------")
    (area_intersection, _) = np.histogram(intersection, range=(0, numClass), bins=numClass)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, range=(0, numClass), bins=numClass)
    (area_lab, _) = np.histogram(imLab, range=(0, numClass), bins=numClass)
    area_union = area_pred + area_lab - area_intersection

    # #print(area_pred)
    # print(area_union)
    # print(area_intersection)
    # print("--------------------------")
    return [area_intersection, area_union]


def update_IoU(preds, labels, counter, counter_no_reset, numClass, batch_size, step, save_every):
    if step % save_every == 0:
        counter[:] = 0

    for i in xrange(batch_size):
        area_intersection, area_union = intersectionAndUnion(preds[i], labels[i], numClass)

        counter[0] = area_intersection
        counter[1] = area_union

        counter_no_reset[0] += area_intersection
        counter_no_reset[1] += area_union

    return (counter, counter_no_reset)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--val-data-list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--gpu-mask", type=str, default=GPU_MASK,
                        help="Comma-separated string for GPU mask.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--first-run", action="store_true",
                        help="First run?")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--val-interval", type=int, default=VAL_INTERVAL,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--conv-lr-multiplier", type=float, default=0.0,
                        help="conv learning rate multiplier")
    parser.add_argument("--fc-w-lr-multiplier", type=float, default=0.0,
                        help="fc_w learning rate multiplier")
    parser.add_argument("--fc-b-lr-multiplier", type=float, default=0.0,
                        help="fc_b learning rate multiplier")
    parser.add_argument("--concat-lr-multiplier", type=float, default=10.0,
                        help="concat learning rate multiplier")
    parser.add_argument("--per-class-train-summary-interval", type=int, default=25,
                        help="save per class summaries every x minibatches")
    return parser.parse_args()


def save(saver, sess, logdir, step):
    '''Save weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    if 'ckpt' in ckpt_path:
        saver.restore(sess, ckpt_path)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    mode = tf.placeholder(tf.bool, shape=())
    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch_train, label_batch_train = reader.dequeue(args.batch_size)

    with tf.name_scope("val_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.val_data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch_val, label_batch_val = reader.dequeue(args.batch_size)

    image_batch = tf.cond(mode, lambda: image_batch_train, lambda: image_batch_val)
    label_batch = tf.cond(mode, lambda: label_batch_train, lambda: label_batch_val)

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if
                   'concat' not in v.name or not args.first_run]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name and 'concat' not in v.name]  # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
    concat_trainable = [v for v in all_trainable if
                        'concat' in v.name]  # only train concat layers -- by zack 07,13,2017
    assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable) + len(concat_trainable))
    assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    # Predictions.
    # net.layers['concat_conv8'] = tf.Print(net.layers['concat_conv8'], conv_trainable, summarize=100)
    # net.layers['concat_conv8'] = tf.Print(net.layers['concat_conv8'], concat_trainable, summarize=100)
    raw_output = net.layers['concat_conv5']
    raw_output_old = net.layers['fc1_voc12']

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack([h, w]), num_classes=args.num_classes,
                               one_hot=False)  # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1, ])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    output_op = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

    correct_pred = tf.equal(output_op, gt)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Pixel-wise softmax loss.
    # loss = []
    accuracy_per_class = []
    #softmax_weights_per_class = tf.constant(LUNA16_softmax_weights, dtype=tf.float32)
    for i in xrange(0, args.num_classes):
        curr_class = tf.constant(i, tf.int32)
        # loss.append(
        #     softmax_weights_per_class[i] * 0.8 * tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=gt,
        #                                                                                 weights=tf.where(
        #                                                                                     tf.equal(gt, curr_class),
        #                                                                                     tf.zeros_like(gt),
        #                                                                                     tf.ones_like(gt))))
        accuracy_per_class.append(
            tf.reduce_mean(tf.cast(tf.gather(correct_pred, tf.where(tf.equal(gt, curr_class))), tf.float32)))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction_old = tf.reshape(raw_output_old, [-1, args.num_classes])
    label_proc_old = prepare_label(label_batch, tf.stack(raw_output_old.get_shape()[1:3]), num_classes=args.num_classes,
                               one_hot=False)  # [batch_size, h, w]
    raw_gt_old = tf.reshape(label_proc_old, [-1, ])
    indices_old = tf.squeeze(tf.where(tf.less_equal(raw_gt_old, args.num_classes - 1)), 1)
    gt_old = tf.cast(tf.gather(raw_gt_old, indices_old), tf.int32)
    prediction_old = tf.gather(raw_prediction_old, indices_old)

    # Pixel-wise softmax loss.
    #softmax_weights_per_class = tf.constant(LUNA16_softmax_weights, dtype=tf.float32)
    # for i in xrange(0, args.num_classes):
    #     curr_class = tf.constant(i, tf.int32)
        # loss.append(softmax_weights_per_class[i] * 0.2 * tf.losses.sparse_softmax_cross_entropy(logits=prediction_old,
        #                                                                                         labels=gt_old,
        #                                                                                         weights=tf.where(
        #                                                                                             tf.equal(gt_old,
        #                                                                                                      curr_class),
        #                                                                                             tf.zeros_like(
        #                                                                                                 gt_old),
        #                                                                                             tf.ones_like(
        #                                                                                                 gt_old))))

    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = 0.0 * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_old, labels=gt_old)) \
                   + 0.8 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)) \
                   + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred_concat = tf.expand_dims(raw_output_up, dim=3)

    raw_output_up_old = tf.image.resize_bilinear(raw_output_old, tf.shape(image_batch)[1:3, ])
    raw_output_up_old = tf.argmax(raw_output_up_old, dimension=3)
    pred_resnet = tf.expand_dims(raw_output_up_old, dim=3)

    # Image summary.
    reduced_loss_train = tf.Variable(0, trainable=False, dtype=tf.float32)
    accuracy_train = tf.Variable(0, trainable=False, dtype=tf.float32)
    reduced_loss_val = tf.Variable(0, trainable=False, dtype=tf.float32)
    accuracy_val = tf.Variable(0, trainable=False, dtype=tf.float32)

    reduced_loss_train = tf.cond(mode, lambda: tf.assign(reduced_loss_train, reduced_loss), lambda: reduced_loss_train)
    accuracy_train = tf.cond(mode, lambda: tf.assign(accuracy_train, accuracy), lambda: accuracy_train)
    reduced_loss_val = tf.cond(mode, lambda: reduced_loss_val, lambda: tf.assign(reduced_loss_val, reduced_loss))
    accuracy_val = tf.cond(mode, lambda: accuracy_val, lambda: tf.assign(accuracy_val, accuracy))

    accuracy_per_class_train = []
    accuracy_per_class_val = []
    for i in xrange(0, args.num_classes):
        temp_train_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        temp_val_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        accuracy_per_class_train.append(
            tf.cond(mode, lambda: tf.assign(temp_train_var, accuracy_per_class[i]), lambda: temp_train_var))
        accuracy_per_class_val.append(
            tf.cond(mode, lambda: temp_val_var, lambda: tf.assign(temp_val_var, accuracy_per_class[i])))

    accuracy_output = tf.cond(mode, lambda: accuracy_train, lambda: accuracy_val)
    loss_output = tf.cond(mode, lambda: reduced_loss_train, lambda: reduced_loss_val)
    tf.summary.scalar("Loss", loss_output, collections=['all'])
    tf.summary.scalar("Accuracy", accuracy_output, collections=['all'])

    counter_no_reset = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32)
    counter = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32)

    counter_no_reset_val = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32)
    counter_val = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32)

    counter, counter_no_reset = tf.cond(mode, lambda: tf.py_func(update_IoU, [tf.squeeze(pred_concat, axis=-1),
                                                                              tf.squeeze(label_batch, axis=-1), counter,
                                                                              counter_no_reset, args.num_classes,
                                                                              args.batch_size, step_ph,
                                                                              args.val_interval],
                                                                 [tf.float32, tf.float32]),
                                        lambda: [counter, counter_no_reset])
    counter_val, counter_no_reset_val = tf.cond(mode,
                                                lambda: [counter_val, counter_no_reset_val],
                                                lambda: tf.py_func(update_IoU, [tf.squeeze(pred_concat, axis=-1),
                                                                                tf.squeeze(label_batch, axis=-1),
                                                                                counter_val, counter_no_reset_val,
                                                                                args.num_classes, args.batch_size,
                                                                                step_ph, args.val_interval],
                                                                   [tf.float32, tf.float32]))

    eps = tf.constant(1e-10, dtype=tf.float32)
    IoU_summary = counter[0] / tf.add(eps, counter[1])
    IoU_summary_no_reset = counter_no_reset[0] / tf.add(eps, counter_no_reset[1])
    Val_IoU_summary = counter_val[0] / tf.add(eps, counter_val[1])
    Val_IoU_summary_no_reset = counter_no_reset_val[0] / tf.add(eps, counter_no_reset_val[1])

    mIoU = tf.reduce_mean(IoU_summary)
    mIoU_no_reset = tf.reduce_mean(IoU_summary_no_reset)
    Val_mIoU = tf.reduce_mean(Val_IoU_summary)
    Val_mIoU_no_reset = tf.reduce_mean(Val_IoU_summary_no_reset)

    IoU_summary_output_intermed = tf.cond(mode, lambda: IoU_summary, lambda: Val_IoU_summary)
    IoU_summary_no_reset_output_intermed = tf.cond(mode, lambda: IoU_summary_no_reset, lambda: Val_IoU_summary_no_reset)
    accuracy_per_class_output_intermed = tf.cond(mode, lambda: accuracy_per_class_train, lambda: accuracy_per_class_val)

    class_number = tf.placeholder(tf.int32, shape=())

    IoU_summary_output = tf.gather(IoU_summary_output_intermed, class_number)
    IoU_summary_no_reset_output = tf.gather(IoU_summary_no_reset_output_intermed, class_number)
    accuracy_per_class_output = tf.gather(accuracy_per_class_output_intermed, class_number)

    tf.summary.scalar("IoU per class", IoU_summary_output, collections=['per_class'])
    tf.summary.scalar("IoU (no reset) per class", IoU_summary_no_reset_output, collections=['per_class'])
    tf.summary.scalar("Accuracy per class", accuracy_per_class_output, collections=['per_class'])

    mIoU_output = tf.cond(mode, lambda: mIoU, lambda: Val_mIoU)
    mIoU_no_reset_output = tf.cond(mode, lambda: mIoU_no_reset, lambda: Val_mIoU_no_reset)
    tf.summary.scalar("mIoU", mIoU_output, collections=['all'])
    tf.summary.scalar("mIoU no reset", mIoU_no_reset_output, collections=['all'])

    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary_concat = tf.py_func(decode_labels, [pred_concat, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary_resnet = tf.py_func(decode_labels, [pred_resnet, args.save_num_images, args.num_classes], tf.uint8)
    tf.summary.image('image, gt, resnet, concat',
                     tf.concat(axis=2,
                               values=[images_summary, labels_summary, preds_summary_resnet, preds_summary_concat]),
                     max_outputs=args.save_num_images, collections=['all'])  # Concatenate row-wise.

    all_summary = tf.summary.merge_all('all')
    per_class_summary = tf.summary.merge_all('per_class')
    summary_writer_train = tf.summary.FileWriter(os.path.join(args.snapshot_dir, 'train_all'),
                                                 graph=tf.get_default_graph())
    summary_writer_val = tf.summary.FileWriter(os.path.join(args.snapshot_dir, 'val_all'),
                                               graph=tf.get_default_graph())

    summary_writer_per_class_val = []
    summary_writer_per_class_train = []
    for i in xrange(args.num_classes):
        summary_writer_per_class_train.append(
            tf.summary.FileWriter(os.path.join(args.snapshot_dir, 'train_class_' + str(i)),
                                  graph=tf.get_default_graph()))
        summary_writer_per_class_val.append(
            tf.summary.FileWriter(os.path.join(args.snapshot_dir, 'val_class_' + str(i)),
                                  graph=tf.get_default_graph()))

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)

    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    opt_conv = tf.train.MomentumOptimizer(learning_rate * args.conv_lr_multiplier, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * args.fc_w_lr_multiplier, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * args.fc_b_lr_multiplier, args.momentum)
    opt_concat = tf.train.MomentumOptimizer(learning_rate * args.concat_lr_multiplier, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable + concat_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):(len(conv_trainable) + len(fc_w_trainable)+len(fc_b_trainable))]
    grads_concat = grads[(len(conv_trainable) + len(fc_w_trainable) + len(fc_b_trainable)):]
    
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
    train_op_concat = opt_concat.apply_gradients(zip(grads_concat, concat_trainable))
    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b, train_op_concat)

    # Set up tf session and initialize variables.
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

    # Load variables if the checkpoint is provided.
    inital_step_value = 1
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        if '.ckpt' in args.restore_from:
            load(loader, sess, args.restore_from)
        else:
            load(loader, sess, tf.train.latest_checkpoint(args.restore_from))
            m = re.search(r'\d+$', tf.train.latest_checkpoint(args.restore_from))
            inital_step_value = int(m.group())

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in xrange(inital_step_value, args.num_steps + 1):
        start_time = time.time()

        # mode False -> val, mode True -> train
        if step % args.save_pred_every == 0:
            save(saver, sess, args.snapshot_dir, step)

        if step % args.val_interval == 0:
            feed_dict = {step_ph: step, mode: False, class_number: step % args.num_classes}
            acc, loss_value, mI, mINR, _, _, _, summary_v_this_class, summary_v = sess.run(
                [accuracy_output, loss_output, mIoU_output, mIoU_no_reset_output, accuracy_per_class_output,
                 IoU_summary_output, IoU_summary_no_reset_output, per_class_summary, all_summary], feed_dict=feed_dict)

            summary_writer_val.add_summary(summary_v, step)
            for i, writer in enumerate(summary_writer_per_class_val):
                feed_dict = {step_ph: step, mode: False, class_number: i}
                _, _, _, summary_v_this_class = sess.run(
                    [accuracy_per_class_output,
                     IoU_summary_output, IoU_summary_no_reset_output, per_class_summary],
                    feed_dict=feed_dict)
                writer.add_summary(summary_v_this_class, step)

            duration = time.time() - start_time
            print(
                'step {:d} \t Val_loss = {:.3f}, Val_acc = {:.3f}, Val_mIoU = {:.6f}, Val_mIoU_no_reset = {:.6f}, ({:.3f} sec/step)'.format(
                    step, loss_value, acc, mI, mINR, duration))
        else:
            feed_dict = {step_ph: step, mode: True, class_number: step % args.num_classes}
            acc, loss_value, mI, mINR, _, _, _, summary_t_this_class, summary_t, _ = sess.run(
                [accuracy_output, loss_output, mIoU_output, mIoU_no_reset_output, accuracy_per_class_output,
                 IoU_summary_output, IoU_summary_no_reset_output, per_class_summary, all_summary, train_op_concat],
                feed_dict=feed_dict)

            summary_writer_train.add_summary(summary_t, step)
            if step % args.per_class_train_summary_interval == 0:
                for i, writer in enumerate(summary_writer_per_class_train):
                    feed_dict = {step_ph: step, mode: True, class_number: i}
                    _, _, _, summary_t_this_class = sess.run(
                        [accuracy_per_class_output,
                         IoU_summary_output, IoU_summary_no_reset_output, per_class_summary],
                        feed_dict=feed_dict)
                    writer.add_summary(summary_t_this_class, step)

            duration = time.time() - start_time
            print(
                'step {:d} \t loss = {:.3f}, acc = {:.3f}, mIoU = {:.6f}, mIoU_no_reset = {:.6f}, ({:.3f} sec/step)'.format(
                    step, loss_value, acc, mI, mINR, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
