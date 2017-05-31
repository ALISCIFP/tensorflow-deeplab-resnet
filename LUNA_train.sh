python train.py  \
--data-dir /home/zack/Data/LUNA16/  \
--data-list /home/zack/Data/LUNA16/dataset/train.txt \
--batch-size 5 \
--num-classes 5 \
--not-restore-last \
--input-size '512,512'  \
--snapshot-dir './snapshots_LUNA16_all8_400k/' \
--gpu-mask '1'  \
--num-steps 400000
