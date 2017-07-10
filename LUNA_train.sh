#!/usr/bin/env bash
python train.py  \
--data-dir /home/zack/Data/LUNA16 \
--data-list /home/zack/Data/LUNA16/dataset/train.txt \
--val-data-list /home/zack/Data/LUNA16/dataset/val.txt \
--not-restore-last
