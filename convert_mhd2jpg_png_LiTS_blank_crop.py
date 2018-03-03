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
import scipy.ndimage.measurements

DATA_DIRECTORY = '/mnt/data/LITS/originalDataAll'
OUT_DIRECTORY = '/home/victor/LITS_GTCrop_OriginalResolution'
PX_TO_EXTEND_BOUNDARY = 5

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


def ndarry2jpg_png((data_file, img_gt_file, out_dir, rescale_to_han, px_to_extend_boundary)):
    ftrain = []
    fval = []

    print data_file, img_gt_file

    img = sitk.ReadImage(data_file)
    img_gt = sitk.ReadImage(img_gt_file)

    img = sitk.GetArrayFromImage(img).transpose()
    img_gt = sitk.GetArrayFromImage(img_gt).transpose().astype(np.int)

    _, fn = os.path.split(data_file)
    _, fn_gt = os.path.split(img_gt_file)

    if rescale_to_han:
        img, input_spacing, out_size = rescale(img, output_spacing=[0.6, 0.6, 0.7], bilinear=True)
        img_gt, _, _ = rescale(img_gt, output_spacing=[0.6, 0.6, 0.7], bilinear=False, input_spacing=input_spacing,
                               output_size=out_size)

    img = np.clip(img, -400, 1000)
    img = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    img_gt_merged = np.copy(img_gt)
    img_gt_merged[img_gt_merged != 0] = 1
    bounding_box = scipy.ndimage.measurements.find_objects(img_gt_merged)[0]

    for i in xrange(np.clip((bounding_box[2].start - px_to_extend_boundary), 1, img.shape[2] - 2),
                    np.clip((bounding_box[2].stop + px_to_extend_boundary), 1,
                            img.shape[2] - 2)):  # because of padding!
        img3c = img[np.clip((bounding_box[0].start - px_to_extend_boundary), 0, img.shape[0] - 1):np.clip(
            (bounding_box[0].stop + px_to_extend_boundary), 0, img.shape[0] - 1),
                np.clip((bounding_box[1].start - px_to_extend_boundary), 0, img.shape[1] - 1):np.clip(
                    (bounding_box[1].stop + px_to_extend_boundary), 0, img.shape[1] - 1),
                (i - 1):(i + 2)]
        scipy.misc.imsave(os.path.join(out_dir, "JPEGImages", fn + "_" + str(i) + ".jpg"), img3c)
        cv2.imwrite(os.path.join(out_dir, "PNGImages", fn_gt + "_" + str(i) + ".png"), img_gt[:, :, i])
        out_string = "/JPEGImages/" + fn + "_" + str(i) + ".jpg\t" + "/PNGImages/" + fn_gt + "_" + str(i) + ".png\n"

        if '99' in data_file:
            fval.append(out_string)
        else:
            ftrain.append(out_string)

    return (ftrain, fval)


def main():
    parser = argparse.ArgumentParser(description="LITS nii to jpg-png file converter")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the LITS dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to output the LITS dataset in jpg and png format.")
    parser.add_argument("--px-to-extend-boundary", type=str, default=PX_TO_EXTEND_BOUNDARY,
                        help="Number of pixels to extend bounding box")
    parser.add_argument("--rescale-to-han", action='store_true',
                        help="Rescale to Han")
    args = parser.parse_args()

    vols = sorted(glob.glob(os.path.join(args.data_dir, '*/volume*.nii')))
    segs = sorted(glob.glob(os.path.join(args.data_dir, '*/segmentation*.nii')))

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

    p = multiprocessing.Pool()
    retval = p.map(ndarry2jpg_png,
                   zip(vols, segs, itertools.repeat(args.out_dir, len(vols)),
                       itertools.repeat(args.rescale_to_han, len(vols)),
                       itertools.repeat(args.px_to_extend_boundary, len(vols))))
    p.close()

    # retval = map(ndarry2jpg_png,
    #                zip(vols, segs, itertools.repeat(args.out_dir, len(vols)),
    #                    itertools.repeat(args.rescale_to_han, len(vols)),
    #                    itertools.repeat(args.px_to_extend_boundary, len(vols))))

    list_train = list(itertools.chain.from_iterable([sublist[0] for sublist in retval]))
    list_val = list(itertools.chain.from_iterable([sublist[1] for sublist in retval]))

    with open(os.path.join(args.out_dir, "dataset/train.txt"), 'w') as ftrain:
        ftrain.writelines(list_train)
    with open(os.path.join(args.out_dir, "dataset/val.txt"), 'w') as fval:
        fval.writelines(list_val)

    print "done."


if __name__ == '__main__':
    main()
