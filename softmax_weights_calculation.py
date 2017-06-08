from __future__ import print_function

import argparse
import csv
import glob
import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

FLAGS = None


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def rescale(input_image, size_z_out, size_y_out, size_x_out):
    resampler = sitk.ResampleImageFilter()
    origin = input_image.GetOrigin()
    resampler.SetOutputOrigin(origin)

    direction = input_image.GetDirection()
    resampler.SetOutputDirection(direction)

    spacing = input_image.GetSpacing()
    orig_size = input_image.GetSize()
    resampler.SetOutputSpacing(
        [spacing[0] * (orig_size[0] - 1) / (size_x_out - 1), spacing[1] * (orig_size[1] - 1) / (size_y_out - 1),
         spacing[2] * (orig_size[2] - 1) / (size_z_out - 1)])

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    size = [size_x_out, size_y_out, size_z_out]
    resampler.SetSize(size)
    return resampler.Execute(input_image)


def convert_to(fold_to_excl, label_fname, fnameout, add_lesions=True, rescale_yes=False):
    try:
        os.makedirs(os.path.join(FLAGS.outdir, fnameout + "/subset" + str(fold_to_excl)))
        os.makedirs(os.path.join(FLAGS.outdir, fnameout + "/subset" + str(fold_to_excl) + "/seg"))
        os.makedirs(os.path.join(FLAGS.outdir, fnameout + "/subset" + str(fold_to_excl) + "/img"))
    except:
        pass

    dict = {}
    with open(os.path.join(FLAGS.srcdir, label_fname), "r") as f:
        csvreader = csv.reader(f)
        next(csvreader)  # skip the header

        for line in csvreader:
            if line[0] not in dict:
                dict[line[0]] = [(np.flipud(np.array(line[1:], dtype=np.float32)[:-1]),
                                  np.array(line[1:], dtype=np.float32)[-1] / 2.0)]
            else:
                dict[line[0]].append((np.flipud(np.array(line[1:], dtype=np.float32)[:-1]),
                                      np.array(line[1:], dtype=np.float32)[-1] / 2.0))

    for fname in glob.iglob(os.path.join(FLAGS.srcdir, "subset" + str(fold_to_excl) + "/*.mhd")):
        sitk_seg_image_raw = sitk.ReadImage(os.path.join(FLAGS.srcdir, "seg-lungs-LUNA16/" + fname.split("/")[-1]))
        sitk_image_raw = sitk.ReadImage(fname)

        # classes are: 0 nothing, 1 yellow - left lung, 2 -blue right lung, 3 - cyan, bronchi, 4, lesions.

        if rescale_yes == True:
            sitk_seg_image_raw = rescale(sitk_seg_image_raw, 50, 50, 50)
            sitk_image_raw = rescale(sitk_image_raw, 50, 50, 50)

        image_raw = sitk.GetArrayFromImage(sitk_image_raw)
        seg_image_raw = sitk.GetArrayFromImage(sitk_seg_image_raw).astype(np.uint8)

        seg_image_raw[seg_image_raw >= 3] -= 2

        if add_lesions:
            seg_img_npOrigin = np.array(list(reversed(sitk_seg_image_raw.GetOrigin())))
            seg_img_npSpacing = np.array(list(reversed(sitk_seg_image_raw.GetSpacing())))

            z = image_raw.shape[0]
            y = image_raw.shape[1]
            x = image_raw.shape[2]

            image_grid = np.array(np.meshgrid(np.arange(0, z), np.arange(0, y), np.arange(0, x), indexing='ij'))

            if fname.split("/")[-1].split(".")[0] in dict:
                for entry in dict[fname.split("/")[-1].split(".")[0]]:
                    lesion_origin = entry[0]
                    lesion_radius = entry[1]
                    seg_image_raw[
                        np.linalg.norm(np.multiply(image_grid.transpose() - worldToVoxelCoord(lesion_origin,
                                                                                              seg_img_npOrigin,
                                                                                              seg_img_npSpacing),
                                                   seg_img_npSpacing).transpose(),
                                       ord=2, axis=0) <= lesion_radius] = 4
                    # need the transpose - numpy won't broadcast otherwise
                    # according to a forum post the units are all mm

        filename_mhd = os.path.join(FLAGS.outdir,
                                    fnameout + "/subset" + str(fold_to_excl) + "/img/" + fname.split("/")[-1])
        sitk.WriteImage(sitk.GetImageFromArray(image_raw), filename_mhd)
        filename_mhd_seg = os.path.join(FLAGS.outdir,
                                        fnameout + "/subset" + str(fold_to_excl) + "/seg/" + fname.split("/")[-1])
        sitk.WriteImage(sitk.GetImageFromArray(seg_image_raw), filename_mhd_seg)

        filename = os.path.join(FLAGS.outdir,
                                fnameout + "/subset" + str(fold_to_excl) + "/" + fname.split("/")[-1] + '.tfrecords')
        print('Writing', filename)
        print(np.unique(seg_image_raw))

        writer = tf.python_io.TFRecordWriter(filename)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw.tostring()),
            'seg_image_raw': _bytes_feature(seg_image_raw.tostring())
        }))
        writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'srcdir',
        type=str,
        help='Directory to read the converted result'
    )
    parser.add_argument(
        'outdir',
        type=str,
        help='Directory to write the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()

    for i in range(0, 10):
        convert_to(i, "annotations.csv", 'no_lesion_no_rescale', add_lesions=False, rescale_yes=False)
        convert_to(i, "annotations.csv", 'yes_lesion_no_rescale', add_lesions=True, rescale_yes=False)
        convert_to(i, "annotations.csv", 'no_lesion_yes_rescale', add_lesions=False, rescale_yes=True)
        convert_to(i, "annotations.csv", 'yes_lesion_yes_rescale', add_lesions=True, rescale_yes=True)
