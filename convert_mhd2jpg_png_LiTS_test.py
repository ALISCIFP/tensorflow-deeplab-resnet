#!/usr/bin/env python

# This script belongs to https://github.com/

import argparse
import glob
import math
import os

import SimpleITK as sitk
import numpy as np
import scipy.misc

DATA_DIRECTORY = '/mnt/data/LITS'
OUT_DIRECTORY = "/mnt/data/newLITS"


def rescale(input_image, output_spacing, bilinear=False):
    resampler = sitk.ResampleImageFilter()
    origin = input_image.GetOrigin()
    resampler.SetOutputOrigin(origin)

    direction = input_image.GetDirection()
    resampler.SetOutputDirection(direction)

    spacing = input_image.GetSpacing()
    orig_size = input_image.GetSize()
    resampler.SetOutputSpacing(output_spacing)

    if bilinear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    size = [int(math.ceil(spacing[0] * (orig_size[0] - 1) / output_spacing[0]) + 1),
            int(math.ceil(spacing[1] * (orig_size[1] - 1) / output_spacing[1]) + 1),
            int(math.ceil(spacing[2] * (orig_size[2] - 1) / output_spacing[2]) + 1)]
    resampler.SetSize(size)
    return resampler.Execute(input_image)


def ndarry2jpg_png(data_file, img_gt_file, out_dir, flist):
    img = sitk.ReadImage(data_file)

    img = rescale(img, output_spacing=[0.6, 0.6, 0.6], bilinear=True)

    img = np.clip(sitk.GetArrayFromImage(img), -400, 1000)
    data_path, fn = os.path.split(data_file)

    img_pad = np.concatenate((np.expand_dims(img[:, :, 0], axis=2), img, np.expand_dims(img[:, :, -1], axis=2)), axis=2)

    for i in xrange(0, img.shape[2]):
        img3c = img_pad[:, :, i:i + 3]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        flist.write("/JPEGImages/" + fn + "_" + str(i) + ".jpg\n")


def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/test-volume*.nii')))

    print "converting",
    if not os.path.exists(os.path.join(out_dir, "JPEGImages")):
        os.mkdir(os.path.join(out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(out_dir, "PNGImages")):
        os.mkdir(os.path.join(out_dir, "PNGImages"))
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.mkdir(os.path.join(out_dir, "dataset"))

    ftest = open(os.path.join(out_dir, "dataset/test.txt"), 'w')

    for vol, seg in vols:
        print vol
        ndarry2jpg_png(vol, out_dir, ftest)

    ftest.close()

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
