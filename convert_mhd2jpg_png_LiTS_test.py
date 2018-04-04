#!/usr/bin/env python

# This script belongs to https://github.com/

import argparse
import glob
import itertools
import math
import multiprocessing
import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy.misc

DATA_DIRECTORY = '/mnt/data/LITS/originalDataAll'
OUT_DIRECTORY = '/home/victor/LITS_NoCrop_OriginalResolution'


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


def ndarry2jpg_png((data_file, out_dir, rescale_to_han)):
    ftest = []
    ftest_3D = []

    img = sitk.ReadImage(data_file)

    if rescale_to_han:
        img, _, _ = rescale(img, output_spacing=[1, 1, 2.5], bilinear=True)

    img = sitk.GetArrayFromImage(img).transpose()

    _, fn = os.path.split(data_file)

    img = np.clip(img, -200, 200)
    img = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    print data_file

    img_nii_orig = nib.load(data_file)
    img_nii_out = nib.Nifti1Image(
        img[:, :, 1:img.shape[2] - 1], img_nii_orig.affine,
        header=img_nii_orig.header)
    img_nii_out.set_data_dtype(np.uint8)
    nib.save(img_nii_out, os.path.join(out_dir, "niiout", fn))

    print(img_nii_out.shape)

    out_string_nii = "/niiout/" + fn + "\n"

    ftest_3D.append(out_string_nii)

    for i in xrange(1, img.shape[2] - 1):  # because of padding!
        img3c = img[:, :, (i - 1):(i + 2)]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i - 1) + ".jpg"), img3c)
        out_string = "/JPEGImages/" + fn + "_" + str(i - 1) + ".jpg\n"
        ftest.append(out_string)

    return ftest, ftest_3D


def main():
    parser = argparse.ArgumentParser(description="LITS nii to jpg-png file converter")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the LITS dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to output the LITS dataset in jpg and png format.")
    parser.add_argument("--rescale-to-han", action='store_true',
                        help="Rescale to Han")
    args = parser.parse_args()

    vols = sorted(glob.glob(os.path.join(args.data_dir, 'test-volume*.nii')))

    print "converting", args
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir, "JPEGImages")):
        os.mkdir(os.path.join(args.out_dir, "JPEGImages"))
    if not os.path.exists(os.path.join(args.out_dir, "PNGImages")):
        os.mkdir(os.path.join(args.out_dir, "PNGImages"))
    if not os.path.exists(os.path.join(args.out_dir, "dataset")):
        os.mkdir(os.path.join(args.out_dir, "dataset"))
    if not os.path.exists(os.path.join(args.out_dir, "niiout")):
        os.mkdir(os.path.join(args.out_dir, "niiout"))

    p = multiprocessing.Pool(8)
    retval = p.map(ndarry2jpg_png,
                   zip(vols, itertools.repeat(args.out_dir, len(vols)),
                       itertools.repeat(args.rescale_to_han, len(vols))))
    p.close()

    # retval = map(ndarry2jpg_png,
    #                zip(vols, segs, itertools.repeat(args.out_dir, len(vols)),
    #                    itertools.repeat(args.rescale_to_han, len(vols)),
    #                    itertools.repeat(args.px_to_extend_boundary, len(vols))))

    list_test = list(itertools.chain.from_iterable([sublist[0] for sublist in retval]))
    list_test_3D = list(itertools.chain.from_iterable([sublist[1] for sublist in retval]))

    with open(os.path.join(args.out_dir, "dataset/test.txt"), 'w') as ftest:
        ftest.writelines(list_test)
    with open(os.path.join(args.out_dir, "dataset/test3D.txt"), 'w') as ftest_3D:
        ftest_3D.writelines(list_test_3D)

    print "done."


if __name__ == '__main__':
    main()
