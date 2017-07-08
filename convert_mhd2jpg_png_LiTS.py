import argparse
import glob
import math
import os
import random

import cv2
import nibabel as nib
import numpy as np
import scipy.misc

DATA_DIRECTORY = '/mnt/data/LITS/'
OUT_DIRECTORY = '/mnt/data/LITS/'


def mhd2ndarray(data_file):
    nibimg = nib.load(data_file)
    img = nibimg.get_data()
    return img


def convert(data_dir, out_dir):
    if not os.path.exists(os.path.join(out_dir, "dataset")):
        os.makedirs(os.path.join(out_dir, "dataset"))

    list_of_all_files = zip(sorted(glob.glob(os.path.join(data_dir, "*/volume*.nii"))),
                            sorted(glob.glob(os.path.join(data_dir, "*/segmentation*.nii"))))
    random.shuffle(list_of_all_files)

    with open(os.path.join(data_dir, "dataset/trainLiver.txt"), 'w') as ftrain, \
            open(os.path.join(data_dir, "dataset/valLiver.txt"), 'w') as fval:

        if not os.path.exists(os.path.join(out_dir, "JPEGImagesLiver")):
            os.makedirs(os.path.join(out_dir, "JPEGImagesLiver"))
        if not os.path.exists(os.path.join(out_dir, "PNGImagesLiver")):
            os.makedirs(os.path.join(out_dir, "PNGImagesLiver"))

        for idx, tuple in enumerate(list_of_all_files):
            data_file, img_gt_file = tuple
            img_gt = mhd2ndarray(img_gt_file)
            img_gt[img_gt == 2] = 0
            print(str(np.unique(img_gt)) + ' ' + data_file)

            img = mhd2ndarray(data_file)
            img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=np.min(img))

            for i in xrange(0, img.shape[2]):
                jpegpath = os.path.join("JPEGImagesLiver", data_file.split('/')[-1] + "_" + str(i) + ".png")
                pngpath = os.path.join("PNGImagesLiver", data_file.split('/')[-1] + "_" + str(i) + ".png")
                scipy.misc.imsave(os.path.join(out_dir, jpegpath), img_pad[:, :, i:i + 3])
                cv2.imwrite(os.path.join(out_dir, pngpath), img_gt[:, :, i])
                if idx <= int(math.floor(0.8 * len(list_of_all_files))):
                    ftrain.write("/" + jpegpath + "\t" + "/" + pngpath + "\n")
                else:
                    fval.write("/" + jpegpath + "\t" + "/" + pngpath + "\n")

        with open(os.path.join(data_dir, "dataset/trainTumor.txt"), 'w') as ftrain, \
                open(os.path.join(data_dir, "dataset/valTumor.txt"), 'w') as fval:

            if not os.path.exists(os.path.join(out_dir, "JPEGImagesTumor")):
                os.makedirs(os.path.join(out_dir, "JPEGImagesTumor"))
            if not os.path.exists(os.path.join(out_dir, "PNGImagesTumor")):
                os.makedirs(os.path.join(out_dir, "PNGImagesTumor"))

            for idx, tuple in enumerate(list_of_all_files):
                data_file, img_gt_file = tuple
                img_gt = mhd2ndarray(img_gt_file)
                img_gt[img_gt == 1] = 0
                img_gt[img_gt == 2] = 1
                print(str(np.unique(img_gt)) + ' ' + data_file)

                img = mhd2ndarray(data_file)
                img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=np.min(img))

                for i in xrange(0, img.shape[2]):
                    jpegpath = os.path.join("JPEGImagesTumor",
                                            data_file.split('/')[-1] + "_" + str(i) + ".png")
                    pngpath = os.path.join("PNGImagesTumor", data_file.split('/')[-1] + "_" + str(i) + ".png")
                    scipy.misc.imsave(os.path.join(out_dir, jpegpath), img_pad[:, :, i:i + 3])
                    cv2.imwrite(os.path.join(out_dir, pngpath), img_gt[:, :, i])
                    if idx <= int(math.floor(0.8 * len(list_of_all_files))):
                        ftrain.write("/" + jpegpath + "\t" + "/" + pngpath + "\n")
                    else:
                        fval.write("/" + jpegpath + "\t" + "/" + pngpath + "\n")

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
