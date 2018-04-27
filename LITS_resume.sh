python train.py  \
--data-dir /home/zack/Data/LITSGroundTruthCropOriginalResolution \
--data-list /home/zack/Data/LITSGroundTruthCropOriginalResolution/dataset/train3D.txt \
--val-data-list /home/zack/Data/LITSGroundTruthCropOriginalResolution/dataset/val3D.txt \
--snapshot-dir './snapshots/HDenseUNet2' \
--learning-rate 1.0e-2 \
--random-scale \
--random-mirror \
--restore-from './snapshots/HDenseUNet2' \
--num-steps 237910 
