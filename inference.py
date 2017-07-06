"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
GPU_MASK = '0'
IGNORE_LABEL = 255
NUM_CLASSES = 21
SAVE_DIR = './output/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--data-dir", type=str,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--model-weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--gpu-mask", type=str, default=GPU_MASK,
                        help="Comma-separated string for GPU mask.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
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
    args = get_arguments()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    with open(args.data_list) as f:
        list_of_filenames = [line.rstrip() for line in f]
        num_steps = len(list_of_filenames)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord,
            shuffle=False)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label,
                                                                            dim=0)  # Add one batch dimension.

    image_batch = tf.image.resize_area(image_batch, [512, 512])
    # # Prepare image.
    # img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # # Convert RGB to BGR.
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # # Extract mean.
    # img -= IMG_MEAN
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    # raw_output = net.layers['fc1_voc12']
    raw_output = net.layers['concat_conv6']
    raw_output_up = tf.image.resize_area(raw_output, tf.shape(label_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Set up TF session and initialize variables. 
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if '.ckpt' in args.model_weights:
        load(loader, sess, args.model_weights)
    else:
        load(loader, sess, tf.train.latest_checkpoint(args.model_weights))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Perform inference.
    for step in range(num_steps):
        preds = sess.run([pred])
        msk = decode_labels(preds[0], num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        im.save(args.save_dir + list_of_filenames[step].split("/")[-1] + '_mask.png')

        print('The output file has been saved to {}'.format(
            args.save_dir + list_of_filenames[step].split("/")[-1] + '_mask.png'))

    coord.request_stop()
    coord.join(threads)

    
if __name__ == '__main__':
    main()
