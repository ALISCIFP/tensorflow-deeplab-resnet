python train.py  \
--data-dir /home/zack/Data/ILD_jpg_png/  \
--data-list /home/zack/Data/ILD_jpg_png/dataset/train.txt \
--num-classes 3 \
--batch-size 5 \
--not-restore-last \
--input-size '512,512'  \
--snapshot-dir './snapshots_ILD/' \
--gpu-mask '0,1'  \

