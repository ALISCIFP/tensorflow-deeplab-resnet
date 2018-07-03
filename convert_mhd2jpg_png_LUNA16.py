#!/usr/bin/env python

# This script belongs to https://github.com/

# this script convert LUNA 16 mhd file to RGB-jpg file.

__author__  = "Zengming Shen,Email:szm0219@gmail.com"

import os,glob
import argparse

import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2
import scipy.misc



DATA_DIRECTORY = '/home/zack/Data/LUNA16/'
OUT_DIRECTORY = "/home/zack/Data/LUNA16/"

def mhd2ndarray(data_file):
    itkimg = sitk.ReadImage(data_file)
    img=sitk.GetArrayFromImage(itkimg)
    img = np.transpose(img,(1,2,0))

    return img

def ndarry2jpg_png(data_file,out_dir,subsetIndex,flist):

    data_path,fn = os.path.split(data_file)
    # img_gt_file= data_path+"output/yes_lesion_no_rescale/seg/"+fn
    img_gt_file = data_file.replace("subset"+str(subsetIndex),"output/yes_lesion_no_rescale/subset"+str(subsetIndex)+"/seg")

    img = mhd2ndarray(data_file)
    img_gt = mhd2ndarray(img_gt_file)

    img_pad=np.lib.pad(img, ((0, 0),(0,0),(1,1)), 'constant', constant_values=(-3024, -3024))
    # img_pos = img_pad-img_pad.min()
    # img_pad = img_pos*(255.0/img_pos.max())

    for i in xrange(0,img.shape[2]):
        img3c = img_pad[:,:,i:i+3]
        try:
            scipy.misc.imsave(os.path.join(out_dir+"JPEGImages/subset"+str(subsetIndex),fn+"_"+str(i)+".jpg"), img3c)
        except ValueError:
            print fn
            pass
        # im = Image.fromarray(img3c)
        # im.save(os.path.join(out_dir+"JPEGImages/subset"+str(subsetIndex),fn+"_"+str(i)+"_"+".jpg"))
        cv2.imwrite(os.path.join(out_dir+"PNGImages/subset"+str(subsetIndex),fn+"_"+str(i)+".png"),img_gt[:,:,i])
        flist.write("/JPEGImages/subset"+str(subsetIndex)+"/"+fn+"_"+str(i)+".jpg "+"/PNGImages/subset"+str(subsetIndex)+"/"+fn+"_"+str(i)+".png\n")

def convert(data_dir,out_dir):

    ftrain = open(data_dir + "dataset/train.txt", 'a')

    fval = open(data_dir + "dataset/val.txt", 'w')

    for i in xrange(3,10):
        print "converting subset "+str(i)



        os.chdir(data_dir + "subset" + str(i))
        if not os.path.exists(data_dir + "JPEGImages/subset" + str(i)):
            os.mkdir(data_dir + "JPEGImages/subset" + str(i))
        if not os.path.exists(data_dir + "PNGImages/subset" + str(i)):
            os.mkdir(data_dir + "PNGImages/subset" + str(i))
        for file in glob.glob("*.mhd"):
            if i<8:
                ndarry2jpg_png(os.path.join(data_dir + "subset" + str(i),file), out_dir, i,ftrain)
            else:
                ndarry2jpg_png(os.path.join(data_dir + "subset" + str(i),file), out_dir, i,fval)

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
