#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np
import SimpleITK as sitk
import cv2
import scipy.misc
import nibabel as nb


DATA_DIRECTORY = '/home/zack/Data/LITS/Train/'
OUT_DIRECTORY = "/home/zack/Data/LITS/"



def ndarry2jpg_png(data_file,img_gt_file,out_dir,flist):


    img = nb.load(data_file).get_data()
    img_gt = nb.load(img_gt_file).get_data()
    data_path,fn = os.path.split(data_file)
    data_path,fn_gt = os.path.split(img_gt_file)

    img_pad=np.concatenate((np.expand_dims(img[:,:,0],axis=2),img,np.expand_dims(img[:,:,-1],axis=2)),axis=2)

    for i in xrange(0,img.shape[2]):
        img3c = img_pad[:,:,i:i+3]
        scipy.misc.imsave(os.path.join(out_dir+"JPEGImages",fn+"_"+str(i)+".jpg"),img3c)
        cv2.imwrite(os.path.join(out_dir+"PNGImages",fn_gt+"_"+str(i)+".png"),img_gt[:,:,i])

        flist.write("/JPEGImages/"+fn+"_"+str(i)+".jpg "+"/PNGImages/"+fn_gt+"_"+str(i)+".png\n")
def convert(data_dir, out_dir):
    vols= sorted(glob.glob(data_dir + 'volume*.nii'))
    segs= sorted(glob.glob(data_dir + 'segmentation*.nii'))



    print "converting",
    if not os.path.exists(out_dir + "JPEGImages"):
        os.mkdir(out_dir + "JPEGImages")
    if not os.path.exists(out_dir + "PNGImages"):
        os.mkdir(out_dir + "PNGImages")
    if not os.path.exists(out_dir + "dataset"):
        os.mkdir(out_dir + "dataset")

    ftrain = open(out_dir + "dataset/train.txt", 'w')

    fval = open(out_dir + "dataset/val.txt", 'w')
    i = 0
    for vol, seg in zip(vols, segs):
        print vol,seg
        ndarry2jpg_png(vol,seg, out_dir,ftrain)

    ftrain.close()
    fval.close()




    print "done."








def main():

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="mdh to jpg-png file converter")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to the directory containing the ILD dataset in jpg and png format.")


    args = parser.parse_args()
    convert(args.data_dir,args.out_dir)


if __name__ == '__main__':
    main()
