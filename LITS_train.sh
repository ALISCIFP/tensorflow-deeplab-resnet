python train.py  \
--data-dir /home/zack/Data/paperrescaledLITS/ \
--data-list /home/zack/Data/paperrescaledLITS/dataset/train.txt \
--val-data-list /home/zack/Data/paperrescaledLITS/dataset/val.txt \
--snapshot-dir './snapshots/HanResNet9Slices' \
--gpu-mask '0,1' \
--learning-rate 1e-3 \
--random-scale \
--random-mirror \
--num-steps 237910 \