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

DATA_DIRECTORY = '/home/victor/LITS'
OUT_DIRECTORY = "/home/victor/newLITS"


def rescale(input_image, output_spacing, bilinear=False):
    resampler = sitk.ResampleImageFilter()
    origin = int put_image.GetOrigin()
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


def ndarry2jpg_png((data_file, out_dir)):
    ftest = []
    img = sitk.ReadImage(data_file)

    print data_file

    img = rescale(img, output_spacing=[0.6, 0.6, 0.7], bilinear=True)

    img = np.clip(sitk.GetArrayFromImage(img).transpose(), -400, 1000)
    num_slices = img.shape[2]
    data_path, fn = os.path.split(data_file)

    img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    for i in xrange(0, num_slices):
        img3c = img_pad[:, :, i:i + 3]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        out_string = "/JPEGImages/" + fn + "_" + str(i) + ".jpg\n"
        ftest.append(out_string)

    return ftest


def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/test-volume*.nii')))

    print "converting"
    if not os.path.exists(os.path.join(out_dir, "JPEGImages")):
        os.mkdir(os.path.join(out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.mkdir(os.path.join(out_dir, "dataset"))

    p = multiprocessing.Pool()
    list_test = p.map(ndarry2jpg_png, zip(vols, itertools.repeat(out_dir, len(vols))))
    p.close()

    list_test = list(itertools.chain.from_iterable(list_test))

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
