#!/usr/bin/env python

# This script belongs to https://github.com/

import os,glob
import argparse

import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2


DATA_DIRECTORY = '/home/zack/Data/ILDDataset/output/yes_lesions_no_rescale_merge_yes/'
OUT_DIRECTORY = "/home/zack/Data/ILDDataset/output/yes_lesions_no_rescale_merge_yes/"

def mhd2ndarray(data_file):
    itkimg = sitk.ReadImage(data_file)
    img=sitk.GetArrayFromImage(itkimg)
    img = np.transpose(img,(1,2,0))

    return img

def ndarry2jpg_png(data_file,out_dir):

    data_path,fn = os.path.split(data_file)
    img_gt_file= data_file.replace("img","seg")

    img = mhd2ndarray(data_file)
    print img.shape
    img_gt = mhd2ndarray(img_gt_file)
    print img_gt.shape

    img_pad=np.lib.pad(img, ((0, 0),(0,0),(1,1)), 'constant', constant_values=(0, 0))

    for i in xrange(0,img_pad.shape[2]):
        img3c = img_pad[:,:,i:i+2]
        im = Image.fromarray(img3c)
        im.save(os.path.join(out_dir+"JPEGImages",fn+"_"+str(i)+"_"+".jpg"))
        cv2.imwrite(os.path.join(out_dir+"PNGImages",fn+"_"+str(i)+"_"+".png"),img_gt[:,:,i])

def convert(data_dir,out_dir):
    os.chdir(data_dir + "img")
    print "converting"

    if not os.path.exists(data_dir + "JPEGImages"):
        os.mkdir(data_dir + "JPEGImages")
    if not os.path.exists(data_dir + "PNGImages"):
        os.mkdir(data_dir + "PNGImages")
    for file in glob.glob("*.mhd"):
        ndarry2jpg_png(os.path.join(data_dir + "img",file), out_dir)








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
