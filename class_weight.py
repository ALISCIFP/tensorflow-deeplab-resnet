#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np
import SimpleITK as sitk
import cv2
import scipy.misc
import nibabel as nb



def count_weight(data_dir,count):
    segs = sorted(glob.glob(data_dir + 'segmentation*.nii'))

    for seg in segs:
        print seg
        print count
        ndArry = nb.load(seg).get_data()
        count = count + [(ndArry == 0).sum(),(ndArry == 1).sum(),(ndArry == 2).sum()]

    print count
    print count/count.sum()








def main():
    DATA_DIRECTORY = '/home/zack/Data/LITS/Train/'
    OUT_DIRECTORY = "/home/zack/Data/LITS/"

    C_Num = 3

    count = np.zeros(C_Num, dtype=np.int64)
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="mdh to jpg-png file converter")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to the directory containing the ILD dataset in jpg and png format.")


    args = parser.parse_args()
    count_weight(args.data_dir,count)


if __name__ == '__main__':
    main()
