#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np
import SimpleITK as sitk

DATA_DIRECTORY = '/home/zack/Data/LUNA16/'

def mhd2raw(data_file):
    itkimg = sitk.ReadImage(data_file)
    img=sitk.GetArrayFromImage(itkimg)
    img = np.transpose(img,(1,2,0))
    return img

def mhd2raw_disk(data_dir):

    for i in xrange(10):
        print "converting subset "+str(i)
        os.chdir(data_dir + "subset" + str(i))
        if not os.path.exists(data_dir + "subset" + str(i) + "braw"):
            os.mkdir(data_dir + "subset" + str(i) + "braw")
        for file in glob.glob("*.mhd"):
            img = mhd2raw(file)
            img.tofile(os.path.join(data_dir + "subset" + str(i) + "braw",file))

    os.chdir(data_dir + "seg-lungs-LUNA16")
    print "seg-lungs-LUNA16"

    if not os.path.exists(data_dir + "seg-lungs-LUNA16" + "braw"):
        os.mkdir(data_dir + "seg-lungs-LUNA16" + "braw")
    for file in glob.glob("*.mhd"):
        img = mhd2raw(file)
        img.tofile(os.path.join(data_dir + "seg-lungs-LUNA16" + "braw", file))


def main():

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="mdh to raw file converter")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the LUNA16 dataset.")


    args = parser.parse_args()
    mhd2raw_disk(args.data_dir)


if __name__ == '__main__':
    main()
