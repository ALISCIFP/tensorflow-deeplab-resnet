import argparse
import glob
import os

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

    with open(os.path.join(data_dir, "dataset/testJPEG.txt"), 'w') as ftest:

        if not os.path.exists(os.path.join(out_dir, "JPEGImagesTest")):
            os.makedirs(os.path.join(out_dir, "JPEGImagesTest"))

        for data_file in glob.iglob(os.path.join(data_dir, "*/test-volume*.nii")):
            print("Writing: " + data_file)
            img = mhd2ndarray(data_file)
            img_pad = np.pad(img, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=np.min(img))

            for i in xrange(0, img.shape[2]):
                jpegpath = os.path.join("JPEGImagesTest", data_file.split('/')[-1] + "_" + str(i) + ".jpg")
                scipy.misc.imsave(os.path.join(out_dir, jpegpath), img_pad[:, :, i:i + 3])
                ftest.write("/" + jpegpath + "\n")


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
