#!/usr/bin/env bash
python train.py  \
--data-dir /home/victor/LUNA16 \
--data-list /mnt/data/LUNA16/dataset/train.txt \
--val-data-list /mnt/data/LUNA16/dataset/val.txt \
--not-restore-last
