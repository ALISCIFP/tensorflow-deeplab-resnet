#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np

from scipy import misc


DATA_DIRECTORY = '/home/zack/Data/ILD_jpg_png/'
DATA_LIST_PATH = '/home/zack/Data/ILD_jpg_png/dataset/train.txt'

DATA_DIRECTORY = '/home/zack/Data/newLITS/'
DATA_LIST_PATH = '/home/zack/Data/newLITS/dataset/train.txt'

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
    im_mean =[0.0,0.0,0.0]
    for f in flist:
        fjpg= args.data_dir+f.split("\t")[0]
        if not os.path.exists(fjpg):
            print fjpg
            continue
        jpgnparray = misc.imread(fjpg)
        im_mean += np.mean(np.mean(jpgnparray,axis=0),axis =0)
        count+=1
        if count % 1000 == 0:
            print count

    print im_mean/count





if __name__ == '__main__':
    main()
