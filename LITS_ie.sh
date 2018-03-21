python evaluate_LiTS.py  \
--data-dir /home/zack/Data/paperLITS \
--data-list /home/zack/Data/paperLITS/dataset/val.txt  \
--restore-from '/home/zack/GitHub/tensorflow-resnet-segmentation/snapshots/HanResNet5Slices_320_R960k/model.ckpt-1720000' \
--batch_size 1 \
--gpu-mask '0'  


python LiTS_eval_offical.py  \
--out-dir ./eval_results/LITS_1720k.csv \
--labels /home/zack/Data/LITS/Train/segmentation-99.nii   \
--probs ./eval/niiout/segmentation-99.nii \
