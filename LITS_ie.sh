python evaluate_LiTS.py  \
--data-dir /home/zack/Data/LITS \
--data-list /home/zack/Data/LITS/dataset/val.txt  \
--restore-from './snapshots/LITS4t2_refine_v2_r2/model.ckpt-18000' \
--batch_size 16 \
--gpu-mask '0'  


python LiTS_eval_offical.py  \
--out-dir ./eval_results/LITS4t2_refine_v2_r2_fc1.csv \
--labels /home/zack/Data/LITS/Train/segmentation-99.nii   \
--probs ./eval/niiout/segmentation-99.nii \
