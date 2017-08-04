#!/usr/bin/env python

# This script belongs to https://github.com/

import argparse
import glob
import itertools
import math
import multiprocessing
import os

import SimpleITK as sitk
import numpy as np
import scipy.misc
import scipy.ndimage

DATA_DIRECTORY = '/mnt/data/rbLITS'
OUT_DIRECTORY = "/home/victor/newLITS_reduced_blanks_new"


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
        size = [int(math.ceil(spacing[0] * (orig_size[0] - 1) / output_spacing[0]) + 1),
                int(math.ceil(spacing[1] * (orig_size[1] - 1) / output_spacing[1]) + 1),
                int(math.ceil(spacing[2] * (orig_size[2] - 1) / output_spacing[2]) + 1)]
    else:
        size = output_size

    resampler.SetSize(size)
    return resampler.Execute(input_image), input_spacing, size


def ndarry2jpg_png((data_file, img_gt_file, out_dir)):
    ftest = []
    img = sitk.ReadImage(data_file)
    img_gt = sitk.ReadImage(img_gt_file)

    spacing = img.GetSpacing()
    gt_spacing = img_gt.GetSpacing()

    img, input_spacing, out_size = rescale(img, output_spacing=[0.6, 0.6, 0.7], bilinear=True)
    img_gt, _, _ = rescale(img_gt, output_spacing=[0.6, 0.6, 0.7], bilinear=False, input_spacing=input_spacing,
                           output_size=out_size)

    img = np.clip(sitk.GetArrayFromImage(img).transpose(), -400, 1000)
    num_slices = img.shape[2]
    img_gt = sitk.GetArrayFromImage(img_gt).transpose()
    data_path, fn = os.path.split(data_file)

    img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    num_empty_to_keep = int(math.ceil(((spacing[0] + spacing[1]) / 2.0) / 0.7 * 10))
    assert num_empty_to_keep >= 0

    connected, num_features = scipy.ndimage.measurements.label(img_gt,
                                                               structure=scipy.ndimage.morphology.generate_binary_structure(
                                                                   3, 3))
    (count, bins) = np.histogram(connected, range=(1, num_features + 1), bins=num_features)
    print bins, count, num_features

    liver_component = np.argmax(count) + 1

    for i in xrange(0, num_slices):
        if i >= img_gt.shape[2]:
            print data_file, img_gt_file, spacing, gt_spacing, 'fail_idx!'
            continue

        if not np.any(connected[:, :,
                      max(0, i - num_empty_to_keep):min(num_slices, i + num_empty_to_keep + 1)] == liver_component):
            print i, num_empty_to_keep, data_file, img_gt_file, spacing, gt_spacing, 'fail_empty!'
            continue

        img3c = img_pad[:, :, i:i + 3]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        out_string = "/JPEGImages/" + fn + "_" + str(i) + ".jpg\n"
        ftest.append(out_string)

    return ftest


def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/test-volume*.nii')))
    segs = sorted(glob.glob(os.path.join(data_dir, '*/test-segmentation*.nii')))

    assert len(vols) == len(segs)

    print "converting"
    if not os.path.exists(os.path.join(out_dir, "JPEGImages")):
        os.mkdir(os.path.join(out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.mkdir(os.path.join(out_dir, "dataset"))

    p = multiprocessing.Pool(2)
    retval = p.map(ndarry2jpg_png, zip(vols, segs, itertools.repeat(out_dir, len(vols))))
    p.close()

    list_test = list(itertools.chain.from_iterable([sublist[0] for sublist in retval]))

    with open(os.path.join(out_dir, "dataset/test.txt"), 'w') as ftest:
        ftest.writelines(list_test)

    print "done."


def main():
    parser = argparse.ArgumentParser(description="mdh to jpg-png file converter")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to the directory containing the ILD dataset in jpg and png format.")
    args = parser.parse_args()
    convert(args.data_dir, args.out_dir)


if __name__ == '__main__':
    main()
