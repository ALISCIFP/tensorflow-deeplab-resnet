"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) #VOC2012
# IMG_MEAN = np.array((40.9729668,   42.62135134,  40.93294311), dtype=np.float32) #ILD
IMG_MEAN = np.array((88.89328702, 89.36887475, 88.8973059), dtype=np.float32)  # LUNA16

GPU_MASK = '1'
BATCH_SIZE = 5
DATA_DIRECTORY = '/home/zack/Data/VOC2012/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/train.txt'
VAL_DATA_LIST_PATH = None
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 5
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 5
SAVE_PRED_EVERY = 10
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    # imPred = imPred * (imLab > 0)

    # Compute area intersection:
    #print(np.unique(imPred))
    #print(np.unique(imLab))

    intersection = np.copy(imPred)
    intersection[imPred != imLab] = -1
   # print(np.unique(intersection))
    # print("--------------------------")
    (area_intersection, _) = np.histogram(intersection, range=(0, numClass), bins=numClass)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, range=(0, numClass),  bins=numClass)
    (area_lab, _) = np.histogram(imLab, range=(0, numClass),  bins=numClass)
    area_union = area_pred + area_lab - area_intersection

    # #print(area_pred)
    # print(area_union)
    # print(area_intersection)
    # print("--------------------------")
    return [area_intersection, area_union]


def update_IoU(preds, labels, counter, counter_no_reset, numClass, batch_size, step, save_every):
    for i in xrange(batch_size):
        area_intersection, area_union = intersectionAndUnion(preds[i], labels[i], numClass)

        if step % save_every == 0:
            counter[0] += area_intersection
            counter[1] += area_union
        else:
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
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
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
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
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
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    print(args)
    np.set_printoptions(suppress=True, precision=6)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    mode = tf.placeholder(tf.bool, shape=[])
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

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]  # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
    assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes,
                               one_hot=False)  # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1, ])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    output_op = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

    correct_pred = tf.equal(output_op, gt)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Image summary.

    tf.summary.scalar("loss", reduced_loss)
    tf.summary.scalar("acc", accuracy)
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    counter_no_reset = tf.Variable(tf.zeros([2, args.num_classes]))
    counter = tf.Variable(tf.zeros([2, args.num_classes]))

    counter_no_reset_val = tf.Variable(tf.zeros([2, args.num_classes]))
    counter_val = tf.Variable(tf.zeros([2, args.num_classes]))

    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    counter, counter_no_reset = tf.cond(mode, lambda: tf.py_func(update_IoU, [tf.squeeze(pred, axis=-1), tf.squeeze(label_batch, axis=-1), counter, counter_no_reset, args.num_classes, args.batch_size, step_ph, args.save_pred_every], [tf.float32, tf.float32]), lambda: [counter, counter_no_reset])
    counter_val, counter_no_reset_val = tf.cond(mode,
            lambda: [counter_val, counter_no_reset_val], lambda: tf.py_func(update_IoU, [tf.squeeze(pred, axis=-1), tf.squeeze(label_batch, axis=-1), counter_val, counter_no_reset_val, args.num_classes, args.batch_size, step_ph, args.save_pred_every], [tf.float32, tf.float32]))

    IoU_summary = counter[0] / counter[1]
    IoU_summary_no_reset = counter_no_reset[0] / counter_no_reset[1]
    Val_IoU_summary = counter_val[0] / counter_val[1]
    Val_IoU_summary_no_reset = counter_no_reset_val[0] / counter_no_reset_val[1]

    mIoU = tf.reduce_mean(IoU_summary)
    mIoU_no_reset = tf.reduce_mean(IoU_summary_no_reset)
    Val_mIoU = tf.reduce_mean(Val_IoU_summary)
    Val_mIoU_no_reset = tf.reduce_mean(Val_IoU_summary_no_reset)

    for i in xrange(args.num_classes):
        tf.summary.scalar("IoU, class " + str(i), IoU_summary[i])
        tf.summary.scalar("IoU (no reset), class " + str(i), IoU_summary_no_reset[i])
        tf.summary.scalar("Val IoU, class " + str(i), Val_IoU_summary[i])
        tf.summary.scalar("Val IoU (no reset), class " + str(i), Val_IoU_summary_no_reset[i])
        tf.summary.scalar("mIoU - Val mIoU, class " + str(i), IoU_summary[i] - Val_IoU_summary[i])
        tf.summary.scalar("mIoU  (no reset) - Val mIoU  (no reset), class " + str(i), IoU_summary_no_reset[i] - Val_IoU_summary_no_reset[i])

    tf.summary.scalar("mIoU", mIoU)
    tf.summary.scalar("mIoU  (no reset)", mIoU_no_reset)
    tf.summary.scalar("Val mIoU", Val_mIoU)
    tf.summary.scalar("Val mIoU  (no reset)", Val_mIoU_no_reset)

    tf.summary.scalar("mIoU - Val mIoU", mIoU - Val_mIoU)
    tf.summary.scalar("mIoU  (no reset) - Val mIoU  (no reset)", mIoU_no_reset - Val_mIoU_no_reset)

    tf.summary.image('images',
                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                     max_outputs=args.save_num_images)  # Concatenate row-wise.

    total_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)

    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in xrange(args.num_steps):
        start_time = time.time()

        if step % args.save_pred_every == 0:
            feed_dict = {step_ph: step, mode: False}
            acc, loss_value, mI, mINR, summary = sess.run(
                [accuracy, reduced_loss, Val_mIoU, Val_mIoU_no_reset, total_summary], feed_dict=feed_dict)
            save(saver, sess, args.snapshot_dir, step)

            summary_writer.add_summary(summary, step)
            duration = time.time() - start_time
            print(
                'step {:d} \t loss = {:.3f}, acc = {:.3f}, Val_mIoU = {:.6f}, Val_mIoU_no_reset = {:.6f}, ({:.3f} sec/step)'.format(
                    step, loss_value, acc, mI, mINR, duration))
        else:
            feed_dict = {step_ph: step, mode: True}
            acc, loss_value, mI, mINR, summary, _ = sess.run(
                [accuracy, reduced_loss, mIoU, mIoU_no_reset, total_summary, train_op], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step)
            duration = time.time() - start_time
            print(
                'step {:d} \t loss = {:.3f}, acc = {:.3f}, mIoU = {:.6f}, mIoU_no_reset = {:.6f}, ({:.3f} sec/step)'.format(
                    step, loss_value, acc, mI, mINR, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
