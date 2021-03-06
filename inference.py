"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
GPU_MASK = '0'


DATA_DIRECTORY = '/home/zack/Data/VOC2012/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/test.txt' #Note, in test.txt each line is /test.jpg /t space
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_DIR = './test_VOC2012/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--gpu-mask", type=str, default=GPU_MASK,
                        help="Comma-separated string for GPU mask.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():

    """Create the model and start the evaluation process."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord,
            shuffle = False)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    # img-list = reader.images
    # mIoU
    # pred = tf.reshape(pred, [-1,])
    # gt = tf.reshape(label_batch, [-1,])
    # weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    #
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    image_fns = reader.images.eval(session=sess)
    for image_fn in image_fns:
        start_time = time.time()

        # print (image_fn)

    # Iterate over training steps.
    # for step in range(args.num_steps):
        # Perform inference.
        preds = sess.run(pred)

        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # print (args.save_dir + os.path.split(image_fn)[1].replace('jpg','png'))
        im.save(args.save_dir + os.path.split(image_fn)[1].replace('jpg','png'))

        duration = time.time() - start_time
        print('duration per pred  ({:.3f} sec/step)'.format( duration))
    print('The output file has been saved to {}'.format(args.save_dir))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
