"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
import csv
import glob
import os
import re
from multiprocessing import Process, Queue, Event

import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import tensorflow as tf

from deeplab_resnet import DeepLabResNetModel, ImageReader

IMG_MEAN = np.array((88.89328702, 89.36887475, 88.8973059), dtype=np.float32)  # LUNA16

GPU_MASK = '0'
DATA_DIRECTORY = None
DATA_LIST_PATH = None
IGNORE_LABEL = 255
NUM_CLASSES = 5
BATCH_SIZE = 20
RESTORE_FROM = './snapshotsLUNA16/'

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
    parser.add_argument("--post-processing", type=bool, default=True,
                        help="Post processing enable or disable")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of classes to predict (including background).")
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
    if '.ckpt' in ckpt_path:
        saver.restore(sess, ckpt_path)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    print("Restored model parameters from {}".format(ckpt_path))


def saving_process(queue, event, num_classes, data_dir, post_processing):
    with open('eval/output.csv', 'wb') as logfile:
        csvwriter = csv.DictWriter(logfile, fieldnames=['File', 'IoU Class 0',
                                                        'IoU Class 1', 'IoU Class 2',
                                                        'IoU Class 3', 'IoU Class 4', 'mIoU'
                                                        ])
        csvwriter.writeheader()
        counter_global = np.zeros((2, num_classes))
        dict_of_curr_processing = {}
        dict_of_curr_processing_len = {}

        while not (event.is_set() and queue.empty()):
            key, idx, preds, labels, acc_per_class, acc, num_slices = queue.get()
            if not os.path.exists('eval/output_' + key + '.csv'):
                with open('eval/output_' + key + '.csv', 'wb') as logfile_per_file:
                    csvwriter_per_file = csv.DictWriter(logfile_per_file, fieldnames=['Z Coord', 'IoU Class 0',
                                                                                      'IoU Class 1', 'IoU Class 2',
                                                                                      'IoU Class 3', 'IoU Class 4',
                                                                                      'mIoU',
                                                                                      'Acc Class 0', 'Acc Class 1',
                                                                                      'Acc Class 2',
                                                                                      'Acc Class 3', 'Acc Class 4',
                                                                                      'Total Acc'])
                    csvwriter_per_file.writeheader()

            with open('eval/output_' + key + '.csv', 'ab') as logfile_per_file:
                csvwriter_per_file = csv.DictWriter(logfile_per_file, fieldnames=['Z Coord', 'IoU Class 0',
                                                                                  'IoU Class 1', 'IoU Class 2',
                                                                                  'IoU Class 3', 'IoU Class 4',
                                                                                  'mIoU',
                                                                                  'Acc Class 0', 'Acc Class 1',
                                                                                  'Acc Class 2',
                                                                                  'Acc Class 3', 'Acc Class 4',
                                                                                  'Total Acc'])

                if key not in dict_of_curr_processing:
                    dict_of_curr_processing[key] = np.zeros((num_slices, 512, 512))
                    dict_of_curr_processing_len[key] = 1  # this is correct!
                counter = np.zeros((2, num_classes))

                if post_processing:
                    preds = scipy.ndimage.morphology.binary_erosion(preds)
                    preds = scipy.ndimage.morphology.binary_dilation(preds)

                area_intersection, area_union = intersectionAndUnion(preds, labels[:, :, 0],
                                                                     num_classes)

                counter[0] += area_intersection
                counter[1] += area_union
                counter_global[0] += area_intersection
                counter_global[1] += area_union

                IoU_per_class = area_intersection / (np.spacing(1) + area_union)

                csvwriter_per_file.writerow({'Z Coord': idx, 'IoU Class 0': IoU_per_class[0],
                                             'IoU Class 1': IoU_per_class[1], 'IoU Class 2': IoU_per_class[2],
                                             'IoU Class 3': IoU_per_class[3], 'IoU Class 4': IoU_per_class[4],
                                             'mIoU': np.mean(IoU_per_class),
                                             'Acc Class 0': acc_per_class[0], 'Acc Class 1': acc_per_class[1],
                                             'Acc Class 2': acc_per_class[2],
                                             'Acc Class 3': acc_per_class[3], 'Acc Class 4': acc_per_class[4],
                                             'Total Acc': acc})

                dict_of_curr_processing[key][idx] = preds
                dict_of_curr_processing_len[key] += 1

                IoU_per_class = counter[0] / (np.spacing(1) + counter[1])


                if dict_of_curr_processing_len[key] == num_slices:
                    print("Writing: " + 'eval/mhdout/' + key + '_out.mhd')
                    csvwriter.writerow({'File': key, 'IoU Class 0': IoU_per_class[0],
                                        'IoU Class 1': IoU_per_class[1], 'IoU Class 2': IoU_per_class[2],
                                        'IoU Class 3': IoU_per_class[3], 'IoU Class 4': IoU_per_class[4],
                                        'mIoU': np.mean(IoU_per_class)
                                        })
                    mhd_out = sitk.GetImageFromArray(dict_of_curr_processing[key])
                    path_to_img = glob.glob(data_dir + '/seg-lungs-LUNA16/' + key + '.mhd')
                    assert len(path_to_img) == 1
                    img = sitk.ReadImage(path_to_img[0])
                    mhd_out.SetDirection(img.GetDirection())
                    mhd_out.SetOrigin(img.GetOrigin())
                    mhd_out.SetSpacing(img.GetSpacing())
                    sitk.WriteImage(mhd_out,
                                    'eval/mhdout/' + key + '_out.mhd')
                    del dict_of_curr_processing[key]
                    dict_of_curr_processing_len[key] += 1

        global_IoU_per_class = counter_global[0] / (np.spacing(1) + counter_global[1])
        csvwriter.writerow({'File': 'Global', 'IoU Class 0': global_IoU_per_class[0],
                            'IoU Class 1': global_IoU_per_class[1], 'IoU Class 2': global_IoU_per_class[2],
                            'IoU Class 3': global_IoU_per_class[3], 'IoU Class 4': global_IoU_per_class[4],
                            'mIoU': np.mean(global_IoU_per_class)
                            })


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    try:
        os.makedirs('eval/mhdout')
    except:
        pass

    event_end = Event()
    queue_proc = Queue()
    with open(args.data_list, 'r') as f:
        list_of_all_lines = f.readlines()
        f.seek(0)

        dict = {}
        for line in f:
            if re.match(".*\\/(.*)\\.mhd.*", line).group(1) not in dict:
                dict[re.match(".*\\/(.*)\\.mhd.*", line).group(1)] = []

            dict[re.match(".*\\/(.*)\\.mhd.*", line).group(1)].append(line)

        with tf.Graph().as_default():
            # Create queue coordinator.
            coord = tf.train.Coordinator()

            # Load reader.
            with tf.name_scope("create_inputs"):
                reader = ImageReader(
                    args.data_dir,
                    args.data_list,
                    (512, 512),  # No defined input size.
                    False,  # No random scale.
                    False,  # No random mirror.
                    args.ignore_label,
                    IMG_MEAN,
                    coord,
                    shuffle=False)
            image_batch, label_batch = reader.dequeue(args.batch_size)

            # Create network.
            net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

            # Which variables to load.
            restore_var = tf.global_variables()

            # Predictions.
            raw_output = net.layers['fc1_voc12']
            raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
            raw_output = tf.argmax(raw_output, dimension=3)
            pred = tf.expand_dims(raw_output, dim=3)  # Create 4-d tensor.

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

            proc = Process(target=saving_process, args=(queue_proc, event_end, args.num_classes,
                                                        args.data_dir, args.post_processing))
            proc.start()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            acc_per_class = np.zeros(args.num_classes)

            for sublist in [list_of_all_lines[i:i + args.batch_size] for i in
                            xrange(0, len(list_of_all_lines), args.batch_size)]:
                preds, labels, acc, acc_per_class[0], acc_per_class[1], \
                acc_per_class[2], acc_per_class[3], acc_per_class[4] = sess.run(
                    [raw_output, label_batch, accuracy, accuracy_per_class[0],
                     accuracy_per_class[1], accuracy_per_class[2], accuracy_per_class[3],
                     accuracy_per_class[4]])
                for i, thing in enumerate(sublist):
                    regex_match = re.match(".*\\/(.*)\\.mhd_([0-9]+).*", thing)
                    # print(regex_match.group(1) + ' ' + str(regex_match.group(2)))
                    queue_proc.put((regex_match.group(1), int(regex_match.group(2)), preds[i], labels[i], acc_per_class,
                                    acc, len(dict[regex_match.group(1)])))

            coord.request_stop()
            coord.join(threads)
            event_end.set()
            proc.join()


if __name__ == '__main__':
    main()
