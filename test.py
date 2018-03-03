"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
import glob
import os
import re
from multiprocessing import Process, Queue, Event

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy.ndimage
import tensorflow as tf

from deeplab_resnet import DeepLabResNetModel, ImageReader

IMG_MEAN = np.array((33.43633936, 33.38798846, 33.43324414), dtype=np.float32)  # LITS resmaple 0.6mm

GPU_MASK = '0'
DATA_DIRECTORY = None
DATA_LIST_PATH = None
IGNORE_LABEL = 255
NUM_CLASSES = 3
BATCH_SIZE = 1
RESTORE_FROM = None

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
    parser.add_argument("--post-processing", action="store_true",
                        help="Post processing enable or disable")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of classes to predict (including background).")
    return parser.parse_args()


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


def rescale(input_image, original_image, bilinear=False):
    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetOutputSpacing(original_image.GetSpacing())
    resampler.SetSize(original_image.GetSize())

    if bilinear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    return resampler.Execute(input_image)


def saving_process(queue, event, data_dir, post_processing, restore_from):
    dict_of_curr_processing = {}
    dict_of_curr_processing_len = {}

    while not (event.is_set() and queue.empty()):
        key, idx, preds, num_slices = queue.get()
        if key not in dict_of_curr_processing:
            dict_of_curr_processing[key] = np.zeros((num_slices, preds.shape[0], preds.shape[1]), dtype=np.uint8)
            dict_of_curr_processing_len[key] = 1  # this is correct!

        dict_of_curr_processing[key][idx] = preds
        dict_of_curr_processing_len[key] += 1

        if dict_of_curr_processing_len[key] == num_slices:
            if post_processing:
                preds_liver = np.copy(dict_of_curr_processing[key])
                preds_liver[preds_liver == 2] = 1
                preds_liver = scipy.ndimage.morphology.binary_erosion(preds_liver.astype(np.uint8),
                                                                      np.ones((3, 3, 3), np.uint8), iterations=3)

                preds_lesion = np.copy(dict_of_curr_processing[key])
                preds_lesion[preds_lesion == 1] = 0
                preds_lesion[preds_lesion == 2] = 1
                preds_lesion = scipy.ndimage.morphology.binary_dilation(preds_lesion.astype(np.uint8),
                                                                        np.ones((3, 3, 3), np.uint8), iterations=3)
                dict_of_curr_processing[key] = preds_lesion.astype(np.uint8) + preds_liver.astype(np.uint8)

            try:
                os.mkdir(os.path.join(restore_from, 'niiout'))
            except:
                pass

            fname_out = os.path.join(restore_from, 'niiout/' + key.replace('volume', 'segmentation') + '.nii')
            print("Writing: " + fname_out)
            path_to_img = glob.glob(data_dir + '/*/' + key + '.nii')
            print(path_to_img)
            assert len(path_to_img) == 1

            base_img_sitk = sitk.ReadImage(path_to_img[0])
            base_prediction_sitk = sitk.GetImageFromArray(dict_of_curr_processing[key])
            base_prediction_sitk.SetOrigin(base_img_sitk.GetOrigin())
            base_prediction_sitk.SetDirection(base_img_sitk.GetDirection())
            base_prediction_sitk.SetSpacing([0.6, 0.6, 0.7])
            prediction_rescaled = sitk.GetArrayFromImage(rescale(base_prediction_sitk, base_img_sitk))

            img = nib.load(path_to_img[0])
            nii_out = nib.Nifti1Image(prediction_rescaled.transpose((1, 2, 0)), img.affine, header=img.header)
            nii_out.set_data_dtype(np.uint8)
            nib.save(nii_out, fname_out)
            del dict_of_curr_processing[key]
            dict_of_curr_processing_len[key] += 1


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    try:
        os.makedirs('eval/niiout')
    except:
        pass

    event_end = Event()
    queue_proc = Queue()
    with open(args.data_list, 'r') as f:
        list_of_all_lines = f.readlines()
        f.seek(0)

        dict = {}
        for line in f:
            if re.match(".*\\/(.*)\\.nii.*", line).group(1) not in dict:
                dict[re.match(".*\\/(.*)\\.nii.*", line).group(1)] = []

            dict[re.match(".*\\/(.*)\\.nii.*", line).group(1)].append(line.rsplit()[0])

        with tf.Graph().as_default():
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
                image = tf.cast(reader.image, tf.float32)
            image_batch = tf.expand_dims(image, dim=0)  # Add one batch dimension.

            # Create network.
            net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

            # Which variables to load.
            restore_var = tf.global_variables()

            # Predictions.
            raw_output = net.layers['fc1_voc12']
            raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
            raw_output = tf.argmax(raw_output, dimension=3)

            sess = tf.Session()
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Load weights.
            loader = tf.train.Saver(var_list=restore_var)
            if args.restore_from is not None:
                load(loader, sess, args.restore_from)

            # Start queue threads.
            proc = Process(target=saving_process, args=(queue_proc, event_end,
                                                        args.data_dir, args.post_processing, args.restore_from))
            proc.start()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            for sublist in [list_of_all_lines[i:i + args.batch_size] for i in
                            xrange(0, len(list_of_all_lines), args.batch_size)]:
                preds = sess.run([raw_output])[0]
                for i, thing in enumerate(sublist):
                    regex_match = re.match(".*\\/(.*)\\.nii_([0-9]+).*", thing)
                    # print(regex_match.group(1) + ' ' + str(regex_match.group(2)))
                    queue_proc.put(
                        (regex_match.group(1), int(regex_match.group(2)), preds[i], len(dict[regex_match.group(1)])))

            coord.request_stop()
            coord.join(threads)
            event_end.set()
            proc.join()


if __name__ == '__main__':
    main()
