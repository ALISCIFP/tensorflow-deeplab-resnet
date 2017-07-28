#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np

from scipy import misc
import scipy


DATA_DIRECTORY = '/home/zack/Data/ILD_jpg_png/'
DATA_LIST_PATH = '/home/zack/Data/ILD_jpg_png/dataset/train.txt'

DATA_DIRECTORY = '/home/zack/Data/newLITS'
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
    im_mean =np.zeros((512,512,3),dtype=np.float32)
    im_mean_square =np.zeros((512,512,3),dtype=np.float32)
    for f in flist:
        fjpg= args.data_dir+f.split(" ")[0]

        jpgnparray = misc.imread(fjpg)
        jpgnparray =jpgnparray.astype(np.float32)
        im_mean += jpgnparray
        im_mean_square +=np.power(jpgnparray,2)
        count +=1
        if count % 1000 == 0:
            print count
    flist_test = args.data_list.replace('train','test')
    ftlist = open(flist_test)

    for ft in ftlist:
        fjpg= args.data_dir+ft.strip('\n')

        jpgnparray = misc.imread(fjpg)
        jpgnparray =jpgnparray.astype(np.float32)
        im_mean += jpgnparray
        im_mean_square +=np.power(jpgnparray,2)
        count +=1
        if count % 1000 == 0:
            print count

    im_mean = im_mean/count
    im_mean_square = im_mean_square/count
    im_var= np.sqrt(im_mean_square - np.power(im_mean,2))
    im_var_std = np.sqrt(im_var)
    print im_mean
    print im_mean_square
    print im_var
    print im_var_std
    np.save(DATA_DIRECTORY+'mean',im_mean)
    np.save(DATA_DIRECTORY+'mean_square',im_mean_square)
    np.save(DATA_DIRECTORY+'var',im_var)
    np.save(DATA_DIRECTORY+'var_std',im_var_std)

    scipy.misc.imsave(DATA_DIRECTORY+'mean.jpeg',im_mean)
    scipy.misc.imsave(DATA_DIRECTORY+'mean_square.jpeg',im_mean_square)
    scipy.misc.imsave(DATA_DIRECTORY+'var.jpeg',im_var)
    scipy.misc.imsave(DATA_DIRECTORY+'var_std.jpeg',im_var_std)







if __name__ == '__main__':
    main()
