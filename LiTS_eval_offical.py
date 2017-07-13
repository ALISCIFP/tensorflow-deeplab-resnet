
# coding: utf-8

# # Evaluation script for LITS Challenge

# In[1]:
import os,glob
import argparse
from medpy import metric
from surface import Surface
import glob
import nibabel as nb
import numpy as np
import os

LABELS = './LITS-CHALLENGE/LITS4/segmentation-99.nii'
PROBS = './LITS-CHALLENGE/LITS4tlr/segmentation-99.nii'
OUT_DIRECTORY = './LiTS_eval_results.csv'

# In[2]:

def get_scores(pred,label,vxlspacing):
	volscores = {}

	volscores['dice'] = metric.dc(pred,label)
	volscores['jaccard'] = metric.binary.jc(pred,label)
	volscores['voe'] = 1. - volscores['jaccard']
	volscores['rvd'] = metric.ravd(label,pred)

	if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
		volscores['assd'] = 0
		volscores['msd'] = 0
	else:
		evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
		volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()

		volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

	return volscores


# ## Load Labels and Predictions




# In[4]:




# # Loop through all volumes

# In[5]:
def eval(label,prob,outpath):

    results = []
    loaded_label = nb.load(label)
    loaded_prob = nb.load(prob)

    liver_scores = get_scores(loaded_prob.get_data()>=1,loaded_label.get_data()>=1,loaded_label.header.get_zooms()[:3])
    lesion_scores = get_scores(loaded_prob.get_data()==2,loaded_label.get_data()==2,loaded_label.header.get_zooms()[:3])
    print "Liver dice",liver_scores['dice'], "Lesion dice", lesion_scores['dice']

    results.append([label, liver_scores, lesion_scores])

    #create line for csv file
    outstr = str(label) + ','
    for l in [liver_scores, lesion_scores]:
        for k,v in l.iteritems():
            outstr += str(v) + ','
            outstr += '\n'

    #create header for csv file if necessary
    if not os.path.isfile(outpath):
        headerstr = 'Volume,'
        for k,v in liver_scores.iteritems():
            headerstr += 'Liver_' + k + ','
        for k,v in liver_scores.iteritems():
            headerstr += 'Lesion_' + k + ','
        headerstr += '\n'
        outstr = headerstr + outstr

    #write to file
    f = open(outpath, 'a+')
    f.write(outstr)
    f.close()



def main():

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="mdh to jpg-png file converter")

    parser.add_argument("--labels", type=str, default=LABELS,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--probs", type=str, default=PROBS,
                        help="Path to the directory containing the ILD dataset.")
    parser.add_argument("--out-dir", type=str, default=OUT_DIRECTORY,
                        help="Path to the directory containing the ILD dataset in jpg and png format.")


    args = parser.parse_args()
    eval(args.labels,args.probs,args.out_dir)


if __name__ == '__main__':
    main()


