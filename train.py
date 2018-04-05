"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.
This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
import os
import re
import shutil
import time

import numpy as np
import tensorflow as tf

from deeplab_resnet import ThreeDNetwork, ImageReaderScaling, decode_labels, inv_preprocess

LUNA16_softmax_weights = np.array((0.2, 1.2, 2.2), dtype=np.float32)  # [15020370189   332764489    18465194]

BATCH_SIZE = 1
DATA_DIRECTORY = None
DATA_LIST_PATH = None
VAL_DATA_LIST_PATH = None
IGNORE_LABEL = 255
INPUT_SIZE = '224,224'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 1000000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 5000
VAL_INTERVAL = 5000
SNAPSHOT_DIR = None
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step per GPU")
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
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--first-run", action="store_true",
                        help="first run?")
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
                        help="Run validation every x minibatches")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")

    return parser.parse_args()


def save(saver, sess, logdir, step):
    """Save weights.
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    """
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    """Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    """
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    print(args)

    if args.first_run:
        try:
            shutil.rmtree(args.snapshot_dir)
        except Exception as e:
            print(e)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    with open(os.path.join(args.data_dir, "dataset", "mean3D.txt"), 'r') as f:
        IMG_MEAN = np.array(f.readline().rstrip(), dtype=np.float32)

    with tf.Graph().as_default():
        # tf.set_random_seed(args.random_seed)

        # Create queue coordinator.
        coord = tf.train.Coordinator()
        # Load reader.
        mode = tf.placeholder(tf.bool, shape=())
        step_ph = tf.placeholder(dtype=tf.float32, shape=())

        with tf.name_scope("create_inputs"):
            train_reader = ImageReaderScaling(
                args.data_dir,
                args.data_list,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord,
                num_threads=1)

        with tf.name_scope("val_inputs"):
            val_reader = ImageReaderScaling(
                args.data_dir,
                args.val_data_list,
                input_size,
                args.random_scale,
                args.random_mirror,
                args.ignore_label,
                IMG_MEAN,
                coord,
                num_threads=1)

        # Define loss and optimisation parameters.
        base_lr = tf.constant(args.learning_rate)

        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
        tf.summary.scalar("Learning Rate", learning_rate, collections=['all'])
        opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)

        counter_no_reset = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32,
                                       name='counter_no_reset')
        counter = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32, name='counter')

        counter_no_reset_val = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32,
                                           name='counter_no_reset_val')
        counter_val = tf.Variable(tf.zeros([2, args.num_classes]), trainable=False, dtype=tf.float32,
                                  name='counter_val')

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            image_batch_train, label_batch_train = train_reader.dequeue(args.batch_size)
            image_batch_val, label_batch_val = val_reader.dequeue(args.batch_size)

            image_batch = tf.cond(mode, lambda: image_batch_train, lambda: image_batch_val)
            label_batch = tf.cond(mode, lambda: label_batch_train, lambda: label_batch_val)
            image_batch = tf.concat(tf.split(image_batch, 12, axis=-1), axis=0)

            # Create network.
            net = ThreeDNetwork({'data': image_batch}, is_training=args.is_training,
                                num_classes=args.num_classes)
            # For a small batch size, it is better to keep
            # the statistics of the BN layers (running means and variances)
            # frozen, and to not update the values provided by the pre-trained model.
            # If is_training=True, the statistics will be updated during the training.
            # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
            # if they are presented in var_list of the optimiser definition.

            # Predictions.
            raw_output = tf.squeeze(net.layers['3d_conv2'], axis=0)
            raw_output_old = net.layers['conv2']

            scope.reuse_variables()
            # Which variables to load. Running means and variances are not trainable,
            # thus all_variables() should be restored.
            restore_var = [v for v in tf.global_variables() or not args.first_run]
            all_trainable = [v for v in tf.trainable_variables() if
                             'beta' not in v.name and 'gamma' not in v.name]

            # Predictions: ignoring all predictions with labels greater or equal than n_classes
            raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])

            raw_gt = tf.reshape(label_batch, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
            gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
            prediction = tf.gather(raw_prediction, indices)

            output_op = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

            correct_pred = tf.equal(output_op, gt)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Pixel-wise softmax loss.
            # loss = []
            accuracy_per_class = []
            # softmax_weights_per_class = tf.constant(LUNA16_softmax_weights, dtype=tf.float32)
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
            # label_proc_old = prepare_label(label_batch, tf.stack(raw_output_old.get_shape()[1:3]),
            #                                num_classes=args.num_classes,
            #                                one_hot=False)  # [batch_size, h, w]
            raw_gt_old = tf.reshape(label_batch, [-1, ])
            indices_old = tf.squeeze(tf.where(tf.less_equal(raw_gt_old, args.num_classes - 1)), 1)
            gt_old = tf.cast(tf.gather(raw_gt_old, indices_old), tf.int32)
            prediction_old = tf.gather(raw_prediction_old, indices_old)

            # Pixel-wise softmax loss.
            # softmax_weights_per_class = tf.constant(LUNA16_softmax_weights, dtype=tf.float32)
            # for i in xrange(0, args.num_classes):
            #     curr_class = tf.constant(i, tf.int32)
            #     loss.append(softmax_weights_per_class[i] * 0.2 * tf.losses.sparse_softmax_cross_entropy(logits=prediction_old,
            #                                                                                         labels=gt_old,
            #                                                                                         weights=tf.where(
            #                                                                                             tf.equal(gt_old,
            #                                                                                                      curr_class),
            #                                                                                             tf.zeros_like(
            #                                                                                                 gt_old),
            #                                                                                             tf.ones_like(
            #                                                                                                 gt_old))))

            l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
            reduced_loss = tf.reduce_mean(
                0.5 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_old, labels=gt_old)) \
                           + tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)) \
                           + tf.add_n(l2_losses)

        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt.apply_gradients(zip(grads, all_trainable))

        # Processed predictions: for visualisation.
        # raw_output_up = tf.image.resize_bilinear(raw_output[0], tf.shape(image_batch)[2:4])
        raw_output_up = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output_up, dim=3)

        image_batch = tf.transpose(image_batch, perm=(3, 1, 2, 0))

        # Image summary.
        reduced_loss_train = tf.Variable(0, trainable=False, dtype=tf.float32)
        accuracy_train = tf.Variable(0, trainable=False, dtype=tf.float32)
        reduced_loss_val = tf.Variable(0, trainable=False, dtype=tf.float32)
        accuracy_val = tf.Variable(0, trainable=False, dtype=tf.float32)

        reduced_loss_train = tf.cond(mode, lambda: tf.assign(reduced_loss_train, reduced_loss),
                                     lambda: reduced_loss_train)
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

        accuracy_per_class_output_intermed = tf.cond(mode, lambda: accuracy_per_class_train,
                                                     lambda: accuracy_per_class_val)

        class_number = tf.placeholder(tf.int32, shape=())

        accuracy_per_class_output = tf.gather(accuracy_per_class_output_intermed, class_number)

        tf.summary.scalar("Accuracy per class", accuracy_per_class_output, collections=['per_class'])

        images_summary = tf.py_func(inv_preprocess, [image_batch[:, :, :, 5:8],
                                                     args.save_num_images, IMG_MEAN],
                                    tf.uint8)
        labels_summary = tf.py_func(decode_labels, [label_batch[:, :, :, 6:7], args.save_num_images, args.num_classes],
                                    tf.uint8)
        preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes],
                                   tf.uint8)
        tf.summary.image('seg output',
                         tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
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

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

        # Set up tf session and initialize variables.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init)

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
                acc, loss_value, _, summary_v_this_class, summary_v = sess.run(
                    [accuracy_output, loss_output, accuracy_per_class_output, per_class_summary, all_summary],
                    feed_dict=feed_dict)

                summary_writer_val.add_summary(summary_v, step)
                summary_writer_per_class_val[step % args.num_classes].add_summary(summary_v_this_class, step)

                duration = time.time() - start_time
                print(
                    'step {:d} \t Val_loss = {:.3f}, Val_acc = {:.3f}, ({:.3f} sec/step)'.format(
                        step, loss_value, acc, duration))
            else:
                feed_dict = {step_ph: step, mode: True, class_number: step % args.num_classes}
                acc, loss_value, _, summary_t_this_class, summary_t, _ = sess.run(
                    [accuracy_output, loss_output, accuracy_per_class_output, per_class_summary, all_summary, train_op],
                    feed_dict=feed_dict)

                summary_writer_train.add_summary(summary_t, step)
                summary_writer_per_class_train[step % args.num_classes].add_summary(summary_t_this_class, step)

                duration = time.time() - start_time
                print(
                    'step {:d} \t loss = {:.3f}, acc = {:.3f}, ({:.3f} sec/step)'.format(
                        step, loss_value, acc, duration))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
