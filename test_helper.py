import os
import shlex
import subprocess

checkpoint_path = "/mnt/data/trainoutput/aug1/snapshots"

ckpt_num_list = []

for sublist in [ckpt_num_list[i:i + 2] for i in
                xrange(0, len(ckpt_num_list), 2)]:
    print(sublist)

    gpu0_proc = subprocess.Popen(shlex.split("python evaluate_LiTS_v2_reduced.py --nii-dir /mnt/data/LITS --data-dir /home/victor/newLITSreducedblankstest \
    --data-list /home/victor/newLITSreducedblankstest/dataset/test.txt --restore-from " + os.path.join(checkpoint_path,
                                                                                                       'snapshots' + str(
                                                                                                           sublist[
                                                                                                               0])) + " \
    --gpu-mask \'0\'"))

    if len(sublist) > 1:
        gpu1_proc = subprocess.Popen(shlex.split("python evaluate_LiTS_v2_reduced.py --nii-dir /mnt/data/LITS --data-dir /home/victor/newLITSreducedblankstest \
                        --data-list /home/victor/newLITSreducedblankstest/dataset/test.txt --restore-from " + os.path.join(
            checkpoint_path, 'snapshots' + str(sublist[1])) + " \
                        --gpu-mask \'1\'"))

    gpu0_proc.wait()
    if len(sublist) > 1:
        gpu1_proc.wait()
