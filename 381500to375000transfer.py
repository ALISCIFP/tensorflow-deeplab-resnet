import glob
import os

import SimpleITK as sitk

if __name__ == '__main__':
    if not os.path.exists("/home/victor/Desktop/output"):
        os.mkdir("/home/victor/Desktop/output")

    for fname in glob.iglob("/home/victor/Desktop/6/*.nii"):
        print fname, os.path.join("/mnt/data/LITS/r1fc1sqw", fname.split("/")[-1]), os.path.join(
            '/home/victor/Desktop/output', fname.split("/")[-1])
        img_381500_np = sitk.GetArrayFromImage(sitk.ReadImage(fname))
        img_375000 = sitk.ReadImage(os.path.join("/mnt/data/LITS/r1fc1sqw", fname.split("/")[-1]))
        img_375000_np = sitk.GetArrayFromImage(img_375000)

        img_375000_np[img_381500_np == 2] = 2

        img_375000_out = sitk.GetImageFromArray(img_375000_np)
        img_375000_out.SetDirection(img_375000.GetDirection())
        img_375000_out.SetSpacing(img_375000.GetSpacing())
        img_375000_out.SetOrigin(img_375000.GetOrigin())

        sitk.WriteImage(img_375000_out, os.path.join('/home/victor/Desktop/output', fname.split("/")[-1]))
