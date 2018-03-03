python train.py  \
--data-dir /home/zack/Data/paperLITS/ \
--data-list /home/zack/Data/paperLITS/dataset/train.txt \
--val-data-list /home/zack/Data/paperLITS/dataset/val.txt \
--snapshot-dir './snapshots/CDenseUNet' \
--learning-rate 2.5e-4 \
--random-scale \
--random-mirror \
--num-steps 237910 \
