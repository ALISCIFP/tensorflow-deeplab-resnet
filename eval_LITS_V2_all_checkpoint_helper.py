import fnmatch
import os
import re
import shlex
import shutil
import subprocess

checkpoint_path = "/mnt/data/trainoutput/aug1/snapshots"

if __name__ == '__main__':
    probs = []
    for root, dirnames, filenames in os.walk(checkpoint_path):
        for filename in fnmatch.filter(filenames, '*.ckpt*'):
            probs.append(os.path.join(root, filename))

    print(probs)

    ckpt_num_list = [int(re.findall(r'\d+', s)[1]) for s in probs]
    print(ckpt_num_list)

    for num in ckpt_num_list:
        if not os.path.exists(os.path.join(checkpoint_path, 'snapshots' + str(num))):
            os.mkdir(os.path.join(checkpoint_path, 'snapshots' + str(num)))
        else:
            shutil.rmtree(os.path.join(checkpoint_path, 'snapshots' + str(num)))
            os.mkdir(os.path.join(checkpoint_path, 'snapshots' + str(num)))

        with open(os.path.join(checkpoint_path, 'snapshots' + str(num), 'checkpoint'), 'w') as list_file:
            list_file.write("model_checkpoint_path: \"model.ckpt-%i\"" % num)

        for s in probs:
            if str(num) in s:
                print(s)
                shutil.copy2(s, os.path.join(checkpoint_path, 'snapshots' + str(num)))

    for sublist in [ckpt_num_list[i:i + 2] for i in
                    xrange(0, len(ckpt_num_list), 2)]:
        print(sublist)

        gpu0_proc = subprocess.Popen(shlex.split("python evaluate_LITS_v2.py --data-dir /home/victor/newLITSreducedblanks \
        --data-list /home/victor/newLITSreducedblanks/dataset/val.txt --restore-from " + os.path.join(checkpoint_path,
                                                                                                      'snapshots' + str(
                                                                                                          sublist[0])) + " \
        --gpu-mask \'0\'"))

        gpu1_proc = subprocess.Popen(shlex.split("python evaluate_LITS_v2.py --data-dir /home/victor/newLITSreducedblanks \
                --data-list /home/victor/newLITSreducedblanks/dataset/val.txt --restore-from " + os.path.join(
            checkpoint_path, 'snapshots' + str(sublist[1])) + " \
                --gpu-mask \'1\'"))

        gpu0_proc.wait()
        gpu1_proc.wait()
