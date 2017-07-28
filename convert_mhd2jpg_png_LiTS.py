#!/usr/bin/env python

# This script belongs to https://github.com/

import argparse
import glob
import math
import os

import SimpleITK as sitk
import cv2
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
    img_gt = sitk.ReadImage(img_gt_file)

    img = rescale(img, output_spacing=[0.6, 0.6, 0.7], bilinear=True)
    img_gt = rescale(img_gt, output_spacing=[0.6, 0.6, 0.7], bilinear=False)

    img = np.clip(sitk.GetArrayFromImage(img).transpose(), -400, 1000)
    img_gt = sitk.GetArrayFromImage(img_gt).transpose()
    data_path, fn = os.path.split(data_file)
    data_path, fn_gt = os.path.split(img_gt_file)

    img_pad = np.concatenate((np.expand_dims(img[:, :, 0], axis=2), img, np.expand_dims(img[:, :, -1], axis=2)), axis=2)

    for i in xrange(0, img.shape[2]):
        img3c = img_pad[:, :, i:i + 3]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        cv2.imwrite(os.path.join(out_dir, "PNGImages", fn_gt + "_" + str(i) + ".png"), img_gt[:, :, i])
        flist.write("/JPEGImages/" + fn + "_" + str(i) + ".jpg\t" + "/PNGImages/" + fn_gt + "_" + str(i) + ".png\n")


def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/volume*.nii')))
    segs = sorted(glob.glob(os.path.join(data_dir, '*/segmentation*.nii')))

    print "converting"
    if not os.path.exists(os.path.join(out_dir, "JPEGImages")):
        os.mkdir(os.path.join(out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(out_dir, "PNGImages")):
        os.mkdir(os.path.join(out_dir, "PNGImages"))
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.mkdir(os.path.join(out_dir, "dataset"))

    ftrain = open(os.path.join(out_dir, "dataset/train.txt"), 'w')
    ftrain_1mm = open(os.path.join(out_dir, "dataset/train1mm.txt"), 'w')
    fval = open(os.path.join(out_dir, "dataset/val.txt"), 'w')

    for vol, seg in zip(vols, segs):
        print vol, seg

        spacing = sitk.ReadImage(vol).GetSpacing()
        print(spacing)
        if any([x > 1.0 for x in spacing[0:2]]):
            ndarry2jpg_png(vol, seg, out_dir, ftrain_1mm)
        else:
            if '99' in vol:
                ndarry2jpg_png(vol, seg, out_dir, fval)
            else:
                ndarry2jpg_png(vol, seg, out_dir, ftrain)

    ftrain.close()
    fval.close()
    ftrain_1mm.close()

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
