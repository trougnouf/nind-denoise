# This script crops a dataset into csxcs with overlap (stride) using crop_img.sh
# typical I/O:
# inputs: datasets/NIND/<set>/NIND_<set>_ISO<val>.<ext>
# outputs: datasets/train/NIND_<cs>_<stride>/<set>/ISO<val>/NIND_<set>_ISO<val>_<xpos>_<ypos>_<stride>.<ext>

import os
import argparse
import subprocess
import math
from multiprocessing import cpu_count
import sys
sys.path.append('..')
from common.libs import utilities

CS = 256
STRIDE = 192
FS_DS_DPATH = os.path.join('..', '..', 'datasets', 'NIND')
CROPPED_DS_DPATH = os.path.join('..', '..', 'datasets', 'cropped')  # no longer used

parser = argparse.ArgumentParser(description='Image cropper with overlap (relies on crop_img.sh)')
parser.add_argument('--cs', default=CS, type=int, help=f'Crop size (default: {CS})')
parser.add_argument('--stride', default=STRIDE, type=int, help=f'Useful crop size (default: {STRIDE})')
parser.add_argument('--dsdir', default=FS_DS_DPATH, type=str, help=f'Input (full-size) dataset directory. Default: {FS_DS_DPATH}')
parser.add_argument('--resdir', default=CROPPED_DS_DPATH, type=str, help=f'Output cropped dataset directory ([resdir]/dsdir_[cs]_[stride]). Default: {CROPPED_DS_DPATH}')
parser.add_argument('--max_threads', default=math.ceil(cpu_count()/2),  type=int, help=f'Maximum number of active threads, default={math.ceil(cpu_count()/2)}')
args = parser.parse_args()

dsdir = utilities.get_leaf(args.dsdir)
resdir = os.path.join(utilities.get_root(args.dsdir), 'cropped', dsdir+'_'+str(args.cs)+'_'+str(args.stride))
todolist = []

def findisoval(fn):
    for split in fn.split('_'):
        if 'ISO' in split:
            return split.split('.')[0]
        elif 'GT' in split in split:
            return fn[fn.find('GT'):].split('.')[0]
        elif 'NOISY' in split:
            return fn[fn.find('NOISY'):].split('.')[0]

sets = os.listdir(args.dsdir)
# structured dataset
if os.path.isdir(os.path.join(args.dsdir, sets[0])):
    for aset in sets:
        isovals=[]
        for image in os.listdir(os.path.join(args.dsdir, aset)):
            inpath=os.path.join(args.dsdir, aset, image)
            isoval=findisoval(image)
            # rename duplicate isoval if any is encountered for easier processing (eg SIDD)
            if isoval in isovals:
                oldval=isoval
                while isoval in isovals:
                    isoval=isoval+'-2'
                newpath = inpath.replace(oldval, isoval)
                os.rename(inpath, inpath.replace(oldval, isoval))
                inpath = newpath
            isovals.append(isoval)
            try:
                outdir=os.path.join(resdir, aset, isoval)
            except TypeError as e:
                print(f'{aset} does not appear to be formatted correctly')
                breakpoint()
            todolist.append(['bash', os.path.join('tools', 'crop_img.sh'), str(args.cs), str(args.stride), inpath, outdir])
# or simple image directory
else:
    for image in os.listdir(args.dsdir):
        inpath = os.path.join(args.dsdir, image)
        outdir = os.path.join(resdir, image[:-4])
        todolist.append(['bash', os.path.join('tools', 'crop_img.sh'), str(args.cs), str(args.stride), inpath, outdir])
# TODO or recursively search for all images
        
processes = set()
for task in todolist:
    #print(task)#dbg
    processes.add(subprocess.Popen(task))
    if len(processes) >= args.max_threads:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
