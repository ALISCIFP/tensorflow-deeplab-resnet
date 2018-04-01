python evaluate_LiTS.py \
--crop-data-dir /home/zack/Data/LITSGroundTruthCropPaperResolution \
--no-crop-data-dir /home/zack/Data/LITSNoCropPaperResolution \
--original-data-dir /home/zack/Data/LITS \
--crop-data-list /home/zack/Data/LITSGroundTruthCropPaperResolution/dataset/val.txt \
--restore-from /home/zack/GitHub/tensorflow-resnet-segmentation/snapshots/HanResNet5Slices_ReLu_320_R1920k \
--gpu-mask '1'  


python LiTS_eval_offical.py  \
--out-dir ./eval_results/LITS_1720k.csv \
--labels /home/zack/Data/LITS/Train/segmentation-99.nii   \
--probs ./eval/niiout/segmentation-99.nii \
