python train.py  \
--data-dir /home/zack/Data/LITSGroundTruthCropOriginalResolution \
--data-list /home/zack/Data/LITSGroundTruthCropOriginalResolution/dataset/train3D.txt \
--val-data-list /home/zack/Data/LITSGroundTruthCropOriginalResolution/dataset/val3D.txt \
--snapshot-dir './snapshots/CDenseUNet' \
--learning-rate 1.0e-2 \
--random-scale \
--random-mirror \
--num-steps 237910 \

