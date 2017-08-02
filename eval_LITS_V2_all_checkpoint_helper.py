import fnmatch
import os
import re
import shlex
import subprocess

checkpoint_path = "/mnt/data/trainoutput/aug1/snapshots"

if __name__ == '__main__':
    probs = []
    for root, dirnames, filenames in os.walk(checkpoint_path):
        for filename in fnmatch.filter(filenames, '*.ckpt*'):
            probs.append(os.path.join(root, filename))

    print(probs)

    cmd = os.getcwd()

    ckpt_num_list = list(set([int(re.findall(r'\d+', s)[1]) for s in probs]))
    print(ckpt_num_list)

    for num in ckpt_num_list:
        if not os.path.exists(os.path.join(checkpoint_path, 'snapshots' + str(num))):
            os.mkdir(os.path.join(checkpoint_path, 'snapshots' + str(num)))


        with open(os.path.join(checkpoint_path, 'snapshots' + str(num), 'checkpoint'), 'w') as list_file:
            list_file.write("model_checkpoint_path: \"model.ckpt-%i\"" % num)

        os.chdir(os.path.join(checkpoint_path, 'snapshots' + str(num)))
        for s in probs:
            if str(num) in s:
                print(s)
                if not os.path.exists(os.path.join(checkpoint_path, 'snapshots' + str(num), s.split("/")[-1])):
                    os.symlink(s, os.path.join(checkpoint_path, 'snapshots' + str(num), s.split("/")[-1]))

    os.chdir(cmd)
    for sublist in [ckpt_num_list[i:i + 2] for i in
                    xrange(0, len(ckpt_num_list), 2)]:
        print(sublist)

        gpu0_proc = subprocess.Popen(shlex.split("python evaluate_LiTS_v2_reduced.py --nii-dir /mnt/data/LITS --data-dir /home/victor/newLITSreducedblanks \
        --data-list /home/victor/newLITSreducedblanks/dataset/val.txt --restore-from " + os.path.join(checkpoint_path,
                                                                                                      'snapshots' + str(
                                                                                                          sublist[0])) + " \
        --gpu-mask \'0\'"))

        if len(sublist) > 1:
            gpu1_proc = subprocess.Popen(shlex.split("python evaluate_LiTS_v2_reduced.py --nii-dir /mnt/data/LITS --data-dir /home/victor/newLITSreducedblanks \
                            --data-list /home/victor/newLITSreducedblanks/dataset/val.txt --restore-from " + os.path.join(
                checkpoint_path, 'snapshots' + str(sublist[1])) + " \
                            --gpu-mask \'1\'"))

        gpu0_proc.wait()
        if len(sublist) > 1:
            gpu1_proc.wait()
