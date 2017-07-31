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
import matplotlib.pyplot as plt

DATA_DIRECTORY = '/home/z003hvsa/Data/LITS'
OUT_DIRECTORY = "/home/z003hvsa/Data/newLITS"



def ndarry2jpg_png(data_file,flist):

    img = sitk.ReadImage(data_file)
    spacing = img.GetSpacing()
    osize = img.GetSize()


    return spacing,osize



def convert(data_dir, out_dir):
    vols = sorted(glob.glob(os.path.join(data_dir, '*/volume*.nii')))
    volsTest = sorted(glob.glob(os.path.join(data_dir+'/Test', '*volume*.nii')))
    vols = vols+volsTest

    ftrain = open(os.path.join(out_dir, "dataset/train.txt"), 'w')
    fval = open(os.path.join(out_dir, "dataset/test.txt"), 'w')
    trainsp =[]
    trainsi =[]
    valsp =[]
    valsi=[]

    for vol in vols:
        print vol


        if not 'test' in vol:
            spacing,osize = ndarry2jpg_png(vol,fval)
            trainsp.append(spacing)
            trainsi.append(osize)


        else:
            spacing,osize = ndarry2jpg_png(vol,ftrain)
            valsp.append(spacing)
            valsi.append(osize)

    ftrain.close()
    fval.close()
    trainspA =np.asarray(trainsp)
    valspA = np.asarray(valsp)

    print trainspA

    print 'train',np.mean(trainspA,axis =1),
    print trainspA

    print 'val' ,np.mean(valspA,axis=1)
    print valspA

    np.save('trainsp',trainspA)
    np.save('valsp',valspA)



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
