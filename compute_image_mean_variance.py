#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np

from scipy import misc
import scipy


DATA_DIRECTORY = '/home/zack/Data/ILD_jpg_png/'
DATA_LIST_PATH = '/home/zack/Data/ILD_jpg_png/dataset/train.txt'

DATA_DIRECTORY = '/home/zack/Data/LITS/'
DATA_LIST_PATH = '/home/zack/Data/LITS/dataset/train.txt'

# DATA_DIRECTORY = '/home/z003hvsa/Data/LiverData_2D_final'
# DATA_LIST_PATH = '/home/z003hvsa/Data/LiverData_2D_final/dataset/train.txt'

def main():

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="mdh to jpg-png file converter")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the directory containing the ILD dataset in jpg and png format.")


    args = parser.parse_args()
    flist = open(args.data_list)
    count = 0
    im_mean =np.zeros((512,512,3),dtype=np.float128)
    im_mean_square =np.zeros((512,512,3),dtype=np.float128)

    for f in flist:
        fjpg= args.data_dir+f.split("\t")[0]
        if not os.path.exists(fjpg):
            print fjpg
            continue
        jpgnparray = misc.imread(fjpg)
        im_mean += jpgnparray
        im_mean_square +=np.power(jpgnparray,2)
        count+=1
        if count % 1000 == 0:
            print count
    im_mean = im_mean/count
    im_mean_square = im_mean_square/count
    im_var = np.sqrt(im_mean_square - np.power(im_mean,2))
    print im_mean
    print im_mean_square
    print im_var
    np.save(DATA_DIRECTORY+'train-mean',im_mean)
    np.save(DATA_DIRECTORY+'train-mean_square',im_mean_square)
    np.save(DATA_DIRECTORY+'train-var',im_var)

    scipy.misc.imsave(DATA_DIRECTORY+'train-mean.jpeg',im_mean)
    scipy.misc.imsave(DATA_DIRECTORY+'train-mean_square.jpeg',im_mean_square)
    scipy.misc.imsave(DATA_DIRECTORY+'train-var.jpeg',im_var)







if __name__ == '__main__':
    main()
