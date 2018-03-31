"""Evaluation script for the HanResNet-5Slice network on the test dataset
   of LiTS 2017 dataset.

This script evaluates the model on 1449 validation images.

"""

from __future__ import print_function

import argparse
import math
import os
import re

import SimpleITK as sitk
import cv2
import nibabel as nib
import numpy as np
import scipy.ndimage
import tensorflow as tf

from deeplab_resnet import DeepLabResNetModel, ImageReader

#IMG_MEAN = np.array((33.43633936, 33.38798846, 33.43324414), dtype=np.float32)  # LITS resmaple 0.6mm
IMG_MEAN = np.array((70.49377469, 70.51345116,  70.66025172), dtype=np.float32) #LITS paper resolution


GPU_MASK = '0'
DATA_DIRECTORY = None
DATA_LIST_PATH = None
IGNORE_LABEL = 255
NUM_CLASSES = 3
BATCH_SIZE = 1
RESTORE_FROM = None


def rescale(input_image, output_spacing, bilinear=False, input_spacing=None, output_size=None):
    resampler = sitk.ResampleImageFilter()
    origin = input_image.GetOrigin()
    resampler.SetOutputOrigin(origin)

    direction = input_image.GetDirection()
    resampler.SetOutputDirection(direction)

    if input_spacing is not None:
        input_image.SetSpacing(input_spacing)

    spacing = input_image.GetSpacing()
    orig_size = input_image.GetSize()
    resampler.SetOutputSpacing(output_spacing)

    if bilinear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    if output_size is None:
        size = [int(math.ceil(spacing[0] * (orig_size[0] - 1) / output_spacing[0])),
                int(math.ceil(spacing[1] * (orig_size[1] - 1) / output_spacing[1])),
                int(math.ceil(spacing[2] * (orig_size[2] - 1) / output_spacing[2]))]
    else:
        size = output_size

    resampler.SetSize(size)
    return resampler.Execute(input_image), input_spacing, size


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--crop-data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--original-data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--no-crop-data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--gpu-mask", type=str, default=GPU_MASK,
                        help="Comma-separated string for GPU mask.")
    parser.add_argument("--crop-data-list", type=str, default=DATA_LIST_PATH,
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


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_mask

    if not os.path.exists(os.path.join(args.restore_from, 'eval/niiout')):
        os.makedirs(os.path.join(args.restore_from, 'eval/niiout'))
    if not os.path.exists(os.path.join(args.restore_from, 'eval/pngout')):
        os.makedirs(os.path.join(args.restore_from, 'eval/pngout'))

    dict_of_curr_processing = {}
    dict_of_curr_processing_len = {}
    dict_of_curr_processing_len_final = {}

    with open(args.crop_data_list, 'r') as f:
        list_of_all_lines = f.readlines()
        f.seek(0)

        dict = {}
        for line in f:
            if re.match(".*\\/(.*)\\.nii.*", line).group(1) not in dict:
                dict[re.match(".*\\/(.*)\\.nii.*", line).group(1)] = []

            dict[re.match(".*\\/(.*)\\.nii.*", line).group(1)].append(line.rsplit()[0])

    with open(os.path.join(args.crop_data_dir, "dataset", "crop_dims.txt"), 'r') as f:
        dict_of_crop_dims = {}
        for line in f:
            line = line.rstrip().split(" ")

            for i in range(len(line[1:])):
                line[i + 1] = int(line[i + 1])

            dict_of_crop_dims[line[0].split(".")[0].replace("volume", "segmentation")] = np.array(line[1:],
                                                                                                  dtype=np.int)

    with tf.Graph().as_default():
        # Create queue coordinator.
        coord = tf.train.Coordinator()

        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                args.crop_data_dir,
                args.crop_data_list,
                None,  # No defined input size.
                False,  # No random scale.
                False,  # No random mirror.
                args.ignore_label,
                IMG_MEAN,
                coord,
                shuffle=False)
            image = tf.cast(reader.image, tf.float32)

        image_batch = tf.expand_dims(image, dim=0)

        # Create network.
        net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['conv24']
        raw_output = tf.argmax(raw_output, axis=3)

        sess = tf.Session()
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for sublist in [list_of_all_lines[i:i + args.batch_size] for i in
                        xrange(0, len(list_of_all_lines), args.batch_size)]:
            preds = sess.run(raw_output)
            for i, thing in enumerate(sublist):
                regex_match = re.match(".*\\/(.*)\\.nii_([0-9]+).*", thing)
                # print(regex_match.group(1) + ' ' + str(regex_match.group(2)))
                cv2.imwrite(os.path.join(args.restore_from, 'eval/pngout', regex_match.group(1).replace('volume',
                                                                                                        'segmentation') + ".nii_" + regex_match.group(
                    2) + ".png"), preds[i])

                key, idx, preds, num_slices = (
                    regex_match.group(1), int(regex_match.group(2)), preds[i], len(dict[regex_match.group(1)]))
                print(key, idx, num_slices)
                if key not in dict_of_curr_processing:
                    dict_of_curr_processing[key] = np.zeros((num_slices, preds.shape[1], preds.shape[0]),
                                                            dtype=np.uint8)
                    dict_of_curr_processing_len[key] = 0  # this is correct!
                    dict_of_curr_processing_len_final[key] = num_slices

                dict_of_curr_processing[key][idx - dict_of_crop_dims[key][4]] = preds.T
                dict_of_curr_processing_len[key] += 1

        coord.request_stop()
        coord.join(threads)

        for key in dict_of_curr_processing_len.keys():
            if dict_of_curr_processing_len_final[key] == dict_of_curr_processing_len[key]:
                output = dict_of_curr_processing[key]

                if args.post_processing:
                    preds_liver = np.copy(output)
                    preds_liver[preds_liver == 2] = 1
                    preds_liver = scipy.ndimage.morphology.binary_erosion(preds_liver.astype(np.uint8),
                                                                          np.ones((3, 3, 3), np.uint8), iterations=5)

                    preds_lesion = np.copy(output)
                    preds_lesion[preds_lesion == 1] = 0
                    preds_lesion[preds_lesion == 2] = 1
                    preds_lesion = scipy.ndimage.morphology.binary_dilation(preds_lesion.astype(np.uint8),
                                                                            np.ones((3, 3, 3), np.uint8), iterations=5)
                    output = preds_lesion.astype(np.uint8) + preds_liver.astype(np.uint8)

                fname_out = os.path.join(args.restore_from,
                                         'eval/niiout/' + key.replace('volume', 'segmentation') + '.nii')
                print("Writing: " + fname_out)

                path_to_img = os.path.join(args.no_crop_data_dir, "niiout", key + '.nii')
                img = nib.load(path_to_img)

                path_to_img_original = os.path.join(args.original_data_dir, key + '.nii')
                img_original_sitk = sitk.ReadImage(path_to_img_original)

                output = output.T
                print(output.shape, img.shape, dict_of_crop_dims[key])

                output = np.pad(output,
                                ((dict_of_crop_dims[key][0], np.maximum(img.shape[0] - dict_of_crop_dims[key][1], 0)),
                                 (dict_of_crop_dims[key][2], np.maximum(img.shape[1] - dict_of_crop_dims[key][3], 0)),
                                 (dict_of_crop_dims[key][4], np.maximum(img.shape[2] - dict_of_crop_dims[key][5], 0))),
                                'constant', constant_values=(0, 0))
                print(output.shape, img.shape[2] - dict_of_crop_dims[key][5])

                output = output.T
                output_sitk = sitk.GetImageFromArray(output)
                output_sitk.SetOrigin(img_original_sitk.GetOrigin())
                output_sitk.SetDirection(img_original_sitk.GetDirection())
                output_sitk.SetSpacing([1, 1, 2.5])
                print(output_sitk.GetSize())

                output_sitk, _, _ = rescale(output_sitk, output_spacing=img_original_sitk.GetSpacing(), bilinear=False,
                                            input_spacing=[1, 1, 2.5], output_size=img_original_sitk.GetSize())

                output = sitk.GetArrayFromImage(output_sitk).transpose()
                print(output.shape)

                nii_out = nib.Nifti1Image(output, img.affine, header=img.header)
                nii_out.set_data_dtype(np.uint8)
                nib.save(nii_out, fname_out)
            else:
                print("fail!", key, dict_of_curr_processing_len_final[key], dict_of_curr_processing_len[key])


if __name__ == '__main__':
    main()
