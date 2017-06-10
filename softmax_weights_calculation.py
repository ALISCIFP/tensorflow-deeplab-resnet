from __future__ import print_function

import argparse
import csv
import glob
import itertools
import os
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np

FLAGS = None

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def convert_to(args):
    fold_to_excl, label_fname, add_lesions = args
    print(str(fold_to_excl) + ' ' + str(add_lesions))

    counter = np.zeros(FLAGS.numClasses, dtype=np.int64)

    dict = {}
    with open(os.path.join(FLAGS.srcdir, label_fname), "r") as f:
        csvreader = csv.reader(f)
        next(csvreader)  # skip the header

        for line in csvreader:
            if line[0] not in dict:
                dict[line[0]] = [(np.flipud(np.array(line[1:], dtype=np.float64)[:-1]),
                                  np.array(line[1:], dtype=np.float64)[-1] / 2.0)]
            else:
                dict[line[0]].append((np.flipud(np.array(line[1:], dtype=np.float64)[:-1]),
                                      np.array(line[1:], dtype=np.float64)[-1] / 2.0))

    for fname in glob.iglob(os.path.join(FLAGS.srcdir, "subset" + str(fold_to_excl) + "/*.mhd")):
        sitk_seg_image_raw = sitk.ReadImage(os.path.join(FLAGS.srcdir, "seg-lungs-LUNA16/" + fname.split("/")[-1]))

        # classes are: 0 nothing, 1 yellow - left lung, 2 -blue right lung, 3 - cyan, bronchi, 4, lesions.

        seg_image_raw = sitk.GetArrayFromImage(sitk_seg_image_raw).astype(np.uint8)

        seg_image_raw[seg_image_raw >= 3] -= 2

        if add_lesions:
            seg_img_npOrigin = np.array(list(reversed(sitk_seg_image_raw.GetOrigin())))
            seg_img_npSpacing = np.array(list(reversed(sitk_seg_image_raw.GetSpacing())))

            z = seg_image_raw.shape[0]
            y = seg_image_raw.shape[1]
            x = seg_image_raw.shape[2]

            image_grid = np.array(np.meshgrid(np.arange(0, z), np.arange(0, y), np.arange(0, x), indexing='ij'))

            if ".".join(fname.split("/")[-1].split(".")[:-1]) in dict:
                for entry in dict[".".join(fname.split("/")[-1].split(".")[:-1])]:
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

        print('Writing', fname)
        print(np.unique(seg_image_raw))
        (area_pred, bins) = np.histogram(seg_image_raw, range=(0, 5), bins=5)
        counter += area_pred
        print(area_pred)
        print(bins)

    return counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'srcdir',
        type=str,
        help='Directory to read the converted result'
    )
    parser.add_argument(
        'numClasses',
        type=int,
        help='Directory to read the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()

    p = Pool()
    output = p.map(convert_to, list(itertools.product(range(8), ["annotations.csv"], [True, False])))

    result = np.zeros(FLAGS.numClasses, dtype=np.int64)
    for a in output:
        print(a)
        result += a

    print(result)
    p.close()
