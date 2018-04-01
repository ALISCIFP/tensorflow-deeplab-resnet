python train.py  \
--data-dir /home/zack/Data/paperLITS/ \
--data-list /home/zack/Data/paperLITS/dataset/train.txt \
--val-data-list /home/zack/Data/paperLITS/dataset/val.txt \
--restore-from './snapshots/HanResNet5Slices_ReLu_320_R1920k' \
--snapshot-dir './snapshots/HanResNet5Slices_ReLu_320_R1920k' \
--gpu-mask '0' \
--batch-size 5 \
--learning-rate 2.5e-4 \
--input-size '320,320' \
--random-scale \
--random-mirror \
--num-steps 3840000 \

