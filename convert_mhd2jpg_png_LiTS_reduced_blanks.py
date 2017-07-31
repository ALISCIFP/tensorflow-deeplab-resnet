#!/usr/bin/env python

# This script belongs to https://github.com/

import argparse
import glob
import itertools
import math
import multiprocessing
import os

import SimpleITK as sitk
import cv2
import numpy as np
import scipy.misc

DATA_DIRECTORY = '/home/victor/newLITS'
OUT_DIRECTORY = "/home/victor/newLITS_reduced_blanks"


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
    ftrain = []
    fval = []
    ftrain_1mm = []
    img = sitk.ReadImage(data_file)
    img_gt = sitk.ReadImage(img_gt_file)

    spacing = img.GetSpacing()
    gt_spacing = img_gt.GetSpacing()
    if any([x == 1.0 for x in spacing[0:2]]):
        print(data_file, img_gt_file, spacing, 'fail!')
    else:
        print data_file, img_gt_file, spacing

    img, input_spacing, out_size = rescale(img, output_spacing=[0.6, 0.6, 0.7], bilinear=True)
    img_gt, _, _ = rescale(img_gt, output_spacing=[0.6, 0.6, 0.7], bilinear=False, input_spacing=input_spacing,
                           output_size=out_size)

    img = np.clip(sitk.GetArrayFromImage(img).transpose(), -400, 1000)
    num_slices = img.shape[2]
    img_gt = sitk.GetArrayFromImage(img_gt).transpose()
    data_path, fn = os.path.split(data_file)
    data_path, fn_gt = os.path.split(img_gt_file)

    img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    num_empty_to_keep = int(math.ceil(((spacing[0] + spacing[1]) / 2.0) / 0.7 * 10))
    assert num_empty_to_keep >= 0

    for i in xrange(0, num_slices):
        if i >= img_gt.shape[2]:
            print data_file, img_gt_file, spacing, gt_spacing, 'fail_idx!'
            continue

        if np.array_equal(np.unique(img_gt[:, :, i - num_empty_to_keep:i + num_empty_to_keep + 1]), [0]):
            print i, num_empty_to_keep, data_file, img_gt_file, spacing, gt_spacing, 'fail_empty!'
            continue

        img3c = img_pad[:, :, i:i + 3]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        cv2.imwrite(os.path.join(out_dir, "PNGImages", fn_gt + "_" + str(i) + ".png"), img_gt[:, :, i])
        out_string = "/JPEGImages/" + fn + "_" + str(i) + ".jpg\t" + "/PNGImages/" + fn_gt + "_" + str(i) + ".png\n"
        if any([x == 1.0 for x in spacing[0:2]]):
            ftrain_1mm.append(out_string)
        elif '99' in data_file:
            fval.append(out_string)
        else:
            ftrain.append(out_string)

    return (ftrain, fval, ftrain_1mm)


def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/volume*.nii')))
    segs = sorted(glob.glob(os.path.join(data_dir, '*/segmentation*.nii')))

    assert len(vols) == len(segs)

    print "converting"
    if not os.path.exists(os.path.join(out_dir, "JPEGImages")):
        os.mkdir(os.path.join(out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(out_dir, "PNGImages")):
        os.mkdir(os.path.join(out_dir, "PNGImages"))
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.mkdir(os.path.join(out_dir, "dataset"))

    p = multiprocessing.Pool(3)
    retval = p.map(ndarry2jpg_png, zip(vols, segs, itertools.repeat(out_dir, len(vols))))
    p.close()

    list_train = list(itertools.chain.from_iterable([sublist[0] for sublist in retval]))
    list_val = list(itertools.chain.from_iterable([sublist[1] for sublist in retval]))
    list_train_1mm = list(itertools.chain.from_iterable([sublist[2] for sublist in retval]))

    with open(os.path.join(out_dir, "dataset/train.txt"), 'w') as ftrain:
        ftrain.writelines(list_train)
    with open(os.path.join(out_dir, "dataset/train1mm.txt"), 'w') as ftrain_1mm:
        ftrain_1mm.writelines(list_train_1mm)
    with open(os.path.join(out_dir, "dataset/val.txt"), 'w') as fval:
        fval.writelines(list_val)

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
