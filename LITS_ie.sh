python evaluate_LiTS.py \
--data-dir /home/zack/Data/LITSNoCropPaperResolution \
--threed-data-dir /home/zack/Data/LITS/Train \
--data-list /home/zack/Data/LITSNoCropPaperResolution/dataset/val.txt \
--restore-from /home/zack/GitHub/tensorflow-resnet-segmentation/snapshots/HanResNet5Slices_ReLu_320_R1920k \
--gpu-mask '1'  


python LiTS_eval_offical.py  \
--out-dir ./eval_results/LITS_1720k.csv \
--labels /home/zack/Data/LITS/Train/segmentation-99.nii   \
--probs ./eval/niiout/segmentation-99.nii \
