"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
import csv
import os
import re
import tempfile

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels

# IMG_MEAN = np.array((88.89328702, 89.36887475, 88.8973059), dtype=np.float32)  # LUNA16
IMG_MEAN = np.array((46.02499091,  46.00602707,  45.95747361), dtype=np.float32)  # Liver_Siemens


GPU_MASK = '0'
DATA_DIRECTORY = "/home/z003hvsa/Data/LiverData_2D_final"
DATA_LIST_PATH = "/home/z003hvsa/Data/LiverData_2D_final/dataset/test.txt"
IGNORE_LABEL = 255
NUM_CLASSES = 2
RESTORE_FROM = './snapshots_Liver1gpu/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--gpu-mask", type=str, default=GPU_MASK,
                        help="Comma-separated string for GPU mask.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()


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
    return area_intersection, area_union

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    print(args)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        os.makedirs('eval/imageout')
        os.makedirs('eval/imageout_raw')
        os.makedirs('eval/mhdout')
    except:
        pass

    dict = {}
    with open(args.data_list, 'r') as f, open('eval/output.csv', 'wb') as logfile:
        csvwriter = csv.DictWriter(logfile, fieldnames=['File', 'IoU Class 0',
                                                        'IoU Class 1', 'mIoU'
                                                        ])
        csvwriter.writeheader()
        for line in f:
            if re.match(".*\\/(.*)\\.mhd.*", line).group(1) not in dict:
                dict[re.match(".*\\/(.*)\\.mhd.*", line).group(1)] = []

            dict[re.match(".*\\/(.*)\\.mhd.*", line).group(1)].append(line)

        counter_global = np.zeros((2, args.num_classes))
        step = 0

        for key in dict:
            with tempfile.NamedTemporaryFile(mode='w') as tempf, open('eval/' + key + '.csv',
                                                                      'wb') as logfile_per_file:
                csvwriter_per_file = csv.DictWriter(logfile_per_file, fieldnames=['Z Coord', 'IoU Class 0',
                                                                                  'IoU Class 1', 'mIoU',
                                                                                  'Acc Class 0', 'Acc Class 1',
                                                                                  'Total Acc'])
                csvwriter_per_file.writeheader()

                tempf.writelines(dict[key])
                tempf.flush()

                prediction_out = np.zeros((len(dict[key]), 512, 512))

                with tf.Graph().as_default():
                    # Create queue coordinator.
                    coord = tf.train.Coordinator()

                    # Load reader.
                    with tf.name_scope("create_inputs"):
                        reader = ImageReader(
                            args.data_dir,
                            tempf.name,
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

                    # Create network.
                    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

                    # Which variables to load.
                    restore_var = tf.global_variables()

                    # Predictions.
                    raw_output = net.layers['fc1_voc12']
                    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
                    raw_output = tf.argmax(raw_output, dimension=3)
                    image_output_raw = tf.image.encode_png(tf.cast(tf.transpose(raw_output, (1, 2, 0)), tf.uint8))
                    pred = tf.expand_dims(raw_output, dim=3)  # Create 4-d tensor.
                    image_output = tf.image.encode_png(
                        tf.squeeze(tf.py_func(decode_labels, [pred, 1, args.num_classes], tf.uint8), axis=0))

                    # mIoU
                    pred = tf.reshape(pred, [-1, ])
                    gt = tf.reshape(label_batch, [-1, ])
                    # weights = tf.cast(tf.less_equal(gt, args.num_classes - 1),
                    #                   tf.int32)  # Ignoring all labels greater than or equal to n_classes.
                    correct_pred = tf.equal(tf.cast(pred, tf.uint8), gt)
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                    accuracy_per_class = []
                    for i in xrange(0, args.num_classes):
                        curr_class = tf.constant(i, tf.uint8)
                        accuracy_per_class.append(tf.reduce_mean(
                            tf.cast(tf.gather(correct_pred, tf.where(tf.equal(gt, curr_class))), tf.float32)))

                    sess = tf.Session()
                    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                    # Load weights.
                    loader = tf.train.Saver(var_list=restore_var)
                    if args.restore_from is not None:
                        load(loader, sess, args.restore_from)

                    # Start queue threads.
                    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                    counter = np.zeros((2, args.num_classes))
                    acc_per_class = np.zeros(args.num_classes)

                    for idx, line in enumerate(dict[key]):
                        colored_label_png, raw_preds_png, preds, labels, acc, acc_per_class[0], acc_per_class[1] = sess.run(
                            [image_output, image_output_raw, raw_output, label_batch, accuracy, accuracy_per_class[0],accuracy_per_class[1]])

                        with open('eval/imageout_raw/' + key + "_" + str(idx) + '.png', 'wb') as f:
                            f.write(raw_preds_png)
                        with open('eval/imageout/' + key + "_" + str(idx) + '.png', 'wb') as f:
                            f.write(colored_label_png)

                        area_intersection, area_union = intersectionAndUnion(preds[0], labels[0, :, :, 0],
                                                                             args.num_classes)

                        counter[0] += area_intersection
                        counter[1] += area_union
                        counter_global[0] += area_intersection
                        counter_global[1] += area_union

                        IoU_per_class = area_intersection / (np.spacing(1) + area_union)

                        csvwriter_per_file.writerow({'Z Coord': idx, 'IoU Class 0': IoU_per_class[0],
                                                     'IoU Class 1': IoU_per_class[1],
                                                     'mIoU': np.mean(IoU_per_class),
                                                     'Acc Class 0': acc_per_class[0], 'Acc Class 1': acc_per_class[1], 'Total Acc': acc})

                        step += 1
                        prediction_out[idx] = preds

                    IoU_per_class = counter[0] / (np.spacing(1) + counter[1])
                    csvwriter.writerow({'File': key, 'IoU Class 0': IoU_per_class[0],
                                        'IoU Class 1': IoU_per_class[1],
                                        'mIoU': np.mean(IoU_per_class)
                                        })

                    print("Writing: " + 'eval/mhdout/' + key + '.mhd')
                    sitk.WriteImage(sitk.GetImageFromArray(prediction_out),
                                    'eval/mhdout/' + key + '.mhd')

                    coord.request_stop()
                    coord.join(threads)

        global_IoU_per_class = counter_global[0] / (np.spacing(1) + counter_global[1])
        csvwriter.writerow({'File': 'Global', 'IoU Class 0': global_IoU_per_class[0],
                            'IoU Class 1': global_IoU_per_class[1],
                            'mIoU': np.mean(global_IoU_per_class)
                            })


if __name__ == '__main__':
    main()
