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


def ndarry2jpg_png((data_file, img_gt_file, out_dir, rescale_to_han)):
    ftrain = []
    fval = []
    ftrain_3D = []
    fval_3D = []
    fcrop_dims = []
    fmean = []
    fmean_3D = []

    img = sitk.ReadImage(data_file)
    img_gt = sitk.ReadImage(img_gt_file)

    if rescale_to_han:
        img, input_spacing, out_size = rescale(img, output_spacing=[1, 1, 2.5], bilinear=True)
        img_gt, _, _ = rescale(img_gt, output_spacing=[1, 1, 2.5], bilinear=False, input_spacing=input_spacing,
                               output_size=out_size)

    img = sitk.GetArrayFromImage(img).transpose()
    img_gt = sitk.GetArrayFromImage(img_gt).transpose().astype(np.int)

    _, fn = os.path.split(data_file)
    _, fn_gt = os.path.split(img_gt_file)

    img = np.clip(img, -200, 200)
    img = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    print data_file, img_gt_file

    img_nii_orig = nib.load(data_file)
    img_nii_out = nib.Nifti1Image(
        img[:, :, 1:img.shape[2] - 1], img_nii_orig.affine,
        header=img_nii_orig.header)
    img_nii_out.set_data_dtype(np.uint8)
    nib.save(img_nii_out, os.path.join(out_dir, "niiout", fn))

    img_gt_nii_orig = nib.load(img_gt_file)
    img_gt_nii_out = nib.Nifti1Image(
        img_gt, img_gt_nii_orig.affine,
        header=img_gt_nii_orig.header)
    img_gt_nii_out.set_data_dtype(np.uint8)
    nib.save(img_gt_nii_out, os.path.join(out_dir, "niiout", fn_gt))

    print(img_nii_out.shape, img_gt_nii_out.shape)

    out_string_nii = "/niiout/" + fn + "\t" + "/niiout/" + fn_gt + "\n"

    if '99' in data_file:
        fval_3D.append(out_string_nii)
    else:
        ftrain_3D.append(out_string_nii)
        fmean_3D.append(np.mean(img_nii_out.get_data()))

    fcrop_dims.append(
        fn + " " + str(0) + " " + str(img_gt.shape[0]) + " " +
        str(0) + " " + str(img_gt.shape[1]) + " " +
        str(0) + " " + str(img_gt.shape[2]) + "\n")

    for i in xrange(1, img.shape[2] - 1):  # because of padding!
        img3c = img[:, :, (i - 1):(i + 2)]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i - 1) + ".jpg"), img3c)
        cv2.imwrite(os.path.join(out_dir, "PNGImages", fn_gt + "_" + str(i - 1) + ".png"), img_gt[:, :, i - 1])
        out_string = "/JPEGImages/" + fn + "_" + str(i - 1) + ".jpg\t" + "/PNGImages/" + fn_gt + "_" + str(
            i - 1) + ".png\n"

        if '99' in data_file:
            fval.append(out_string)
        else:
            ftrain.append(out_string)
            fmean.append(
                np.mean(scipy.misc.imread(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i - 1) + ".jpg"))))

    return ftrain, fval, ftrain_3D, fval_3D, fcrop_dims, fmean, fmean_3D


def ndarry2jpg_png_test((data_file, out_dir, rescale_to_han)):
    ftest = []
    ftest_3D = []
    fcrop_dims = []

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

    fcrop_dims.append(
        fn + " " + str(0) + " " + str(img.shape[0]) + " " +
        str(0) + " " + str(img.shape[1]) + " " +
        str(0) + " " + str(img.shape[2] - 2) + "\n")

    return ftest, ftest_3D, fcrop_dims


def main():
    parser = argparse.ArgumentParser(description="LITS nii to jpg-png file converter")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the LITS dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to output the LITS dataset in jpg and png format.")
    parser.add_argument("--rescale-to-han", action='store_true',
                        help="Rescale to Han")
    args = parser.parse_args()

    vols = sorted(glob.glob(os.path.join(args.data_dir, 'volume*.nii')))
    segs = sorted(glob.glob(os.path.join(args.data_dir, 'segmentation*.nii')))

    vols_test = sorted(glob.glob(os.path.join(args.data_dir, 'test-volume*.nii')))

    assert len(vols) == len(segs)

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
                   zip(vols, segs, itertools.repeat(args.out_dir, len(vols)),
                       itertools.repeat(args.rescale_to_han, len(vols))))
    retval_test = p.map(ndarry2jpg_png_test,
                        zip(vols_test, itertools.repeat(args.out_dir, len(vols_test)),
                            itertools.repeat(args.rescale_to_han, len(vols_test))))
    p.close()

    # retval = map(ndarry2jpg_png,
    #                zip(vols, segs, itertools.repeat(args.out_dir, len(vols)),
    #                    itertools.repeat(args.rescale_to_han, len(vols)),
    #                    itertools.repeat(args.px_to_extend_boundary, len(vols))))

    list_train = list(itertools.chain.from_iterable([sublist[0] for sublist in retval]))
    list_val = list(itertools.chain.from_iterable([sublist[1] for sublist in retval]))
    list_train_3D = list(itertools.chain.from_iterable([sublist[2] for sublist in retval]))
    list_val_3D = list(itertools.chain.from_iterable([sublist[3] for sublist in retval]))
    list_crop_dims = list(itertools.chain.from_iterable([sublist[4] for sublist in retval]))
    list_mean = list(itertools.chain.from_iterable([sublist[5] for sublist in retval]))
    list_mean_3D = list(itertools.chain.from_iterable([sublist[6] for sublist in retval]))

    list_mean = np.mean(list_mean)
    list_mean_3D = np.mean(list_mean_3D)

    list_test = list(itertools.chain.from_iterable([sublist[0] for sublist in retval_test]))
    list_test_3D = list(itertools.chain.from_iterable([sublist[1] for sublist in retval_test]))
    list_crop_dims_test = list(itertools.chain.from_iterable([sublist[2] for sublist in retval_test]))

    with open(os.path.join(args.out_dir, "dataset/train.txt"), 'w') as ftrain:
        ftrain.writelines(list_train)
    with open(os.path.join(args.out_dir, "dataset/val.txt"), 'w') as fval:
        fval.writelines(list_val)
    with open(os.path.join(args.out_dir, "dataset/train3D.txt"), 'w') as ftrain_3D:
        ftrain_3D.writelines(list_train_3D)
    with open(os.path.join(args.out_dir, "dataset/crop_dims_train.txt"), 'w') as fcropdims_train:
        fcropdims_train.writelines(list_crop_dims)
    with open(os.path.join(args.out_dir, "dataset/mean.txt"), 'w') as fmean:
        fmean.writelines(list_mean)
    with open(os.path.join(args.out_dir, "dataset/mean3D.txt"), 'w') as fmean_3D:
        fmean_3D.writelines(list_mean_3D)
    with open(os.path.join(args.out_dir, "dataset/val3D.txt"), 'w') as fval_3D:
        fval_3D.writelines(list_val_3D)

    with open(os.path.join(args.out_dir, "dataset/test.txt"), 'w') as ftest:
        ftest.writelines(list_test)
    with open(os.path.join(args.out_dir, "dataset/test3D.txt"), 'w') as ftest_3D:
        ftest_3D.writelines(list_test_3D)
    with open(os.path.join(args.out_dir, "dataset/crop_dims_test.txt"), 'w') as fcropdims_test:
        fcropdims_test.writelines(list_crop_dims_test)

    print "done."


if __name__ == '__main__':
    main()
