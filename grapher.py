# results collection helper
import os
import argparse
import matplotlib.pyplot as plt
import numpy
from random import random
from math import floor
from dataset_torch_3 import sortISOs
import csv
import json

# generate results from trained model with python denoise_dir.py --model_subdir <model>
# python grapher.py --mode keywords --components p2p
# TODO add min date
# history:
#

# Params
parser = argparse.ArgumentParser(description='Grapher')
parser.add_argument('--res_dir', default='results/test', type=str, help='Results directory')
#parser.add_argument('--res_bfn', default='train_result', type=str, help='Result base filename (s.t. [res_bfn]_[epoch].txt)')
parser.add_argument('--save_dir', default='graphs/auto', type=str, help='Where graphs are saved')
parser.add_argument('--nodisplay', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('--xaxis', type=str, default='ISO')
parser.add_argument('--yaxis', type=str, default='SSIM')
parser.add_argument('--components', nargs='+', help='space-separated line components (eg Noisy NIND Artificial BM3D) or keywords (eg: p2p)')
parser.add_argument('--metric', default='ssim', help='Metric shown (ssim, mse)')
#parser.add_argument('--width', default=15, type=int)
#parser.add_argument('--height', default=7, type=int)
parser.add_argument('--run', default=None, type=int, help="Generate a single graph number")
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--nojson', action='store_true')
parser.add_argument('--mode', default='std', help='std, keywords, all')
#parser.add_argument('--blacklist', nargs='*', help='Space separated list of experiments that should be avoided')
args = parser.parse_args()

data = dict()
#data = {label: {resfiles: [] , results: {image: {isoval {ssim: , mse, exp=None}}}
#match labels to fn
#load data
#generate a graph for image

if args.mode == 'std':
    if args.components:
        components = args.components
    else:
        components = ['Noisy', 'NIND:X-T1 (U-Net)', 'NIND (GAN)', 'NIND (cGAN)', 'SIDD (U-Net)', 'BM3D', 'NIND:X-T1+C500D (U-Net)', 'NIND:X-T1+C500D + SIDD (U-Net)', 'NIND:X-T1 ISO6400-only (U-Net)', 'Artificial noise on NIND:X-T1 (U-Net)', 'Reconstruct noise on NIND:X-T1 (U-Net)', 'NIND:X-T1 (Red-Net)']
else:
    #components = ['GT']
    components = ['Noisy input', 'BM3D'] # TODO BM3D should be optional
    for experiment in os.listdir(args.res_dir):
        if args.mode == 'all':
            if experiment == 'GT' or experiment == 'Noisy input' or 'bm3d' in experiment:
                continue
            components.append(experiment)
        elif args.mode == 'keywords':
            for keyword in args.components:
                if keyword in experiment:
                    components.append(experiment)
        else:
            print('Error: unknown mode: '+args.mode)
    print(components)

def gen_markers():
    markers = []
    i = 0
    while len(markers) < len(components):
        markers.append("$%i$" % i)
        i+=1
    return markers

def make_markers_dict(components = components, markers = gen_markers()):
    markersdict = dict()
    i = 0
    for acomp in components:
        markersdict[acomp] = markers[i]
        i += 1
        if i >= len(markers):
            i = 0
    return markersdict

markers = make_markers_dict()


def find_relevant_experiments(component):
    experiments = os.listdir(args.res_dir)
    data[component] = {'resfiles': []}  # check existing if data is needed earlier
    def add_exp_to_data(experiment):
        respath = os.path.join(args.res_dir, experiment, 'res.txt')
        if os.path.isfile(respath):
            data[component]['resfiles'].append(respath)
    if component == 'Noisy':
        return add_exp_to_data('GT')
    for experiment in experiments:
        if 'bm3d' in experiment:
            if component == 'BM3D':
                add_exp_to_data(experiment)
        elif args.mode != 'std':
            if experiment == component:
                add_exp_to_data(experiment)
        elif 'p2p' in experiment:
            if 'not_conditional' in experiment:
                if component == 'NIND (GAN)':
                    add_exp_to_data(experiment)
            else:
                if component == 'NIND (cGAN)':
                    add_exp_to_data(experiment)
        elif 'RedCNN' in experiment:
            if 'Red-Net' in component:
                add_exp_to_data(experiment)
        elif 'ISO6400' in experiment:
            if 'ISO6400' in component:
                add_exp_to_data(experiment)
        elif 'find_noise' in experiment:
            if 'Model noise' in component:
                add_exp_to_data(experiment)
        elif '--yisx' in experiment and '--sigmamax' in experiment:
            if 'Artificial' in component:
                add_exp_to_data(experiment)
        elif 'SIDD' in experiment and 'NIND' in experiment:
            if 'SIDD' in component and 'NIND' in component:
                add_exp_to_data(experiment)
        elif 'SIDD' in experiment and 'NIND' not in experiment:
            if 'SIDD' in component and 'NIND' not in experiment:
                add_exp_to_data(experiment)
        elif 'C500D' in experiment and 'SIDD' not in experiment:
            if 'C500D' in component and 'SIDD' not in component:
                add_exp_to_data(experiment)
        elif 'run_nn.py' in experiment:
            if component == 'NIND:X-T1 (U-Net)':
                add_exp_to_data(experiment)


def parse_resfiles(component):
    data[component]['results'] = {}
    for respath in data[component]['resfiles']:
        with open(respath) as f:
            for res in csv.reader(f):
                image, iso = res[0].split('_')[1:3]
                isoval = iso.split('.')[0]
                newssim, newmse = float(res[1]), float(res[2])
                oldssim, oldmse = 0.0, 1.0
                if image not in data[component]['results']:
                    data[component]['results'][image] = {}
                if isoval in data[component]['results'][image]:
                    oldssim = data[component]['results'][image][isoval]['ssim']
                    oldmse = data[component]['results'][image][isoval]['mse']
                if newssim > oldssim:
                    data[component]['results'][image][isoval] = {'ssim': float(res[1]), 'mse': float(res[2]), 'exp': respath.split('/')[-2]}

for component in components:
    find_relevant_experiments(component)
    parse_resfiles(component)

images = list(data[components[0]]['results'].keys())
for image in images:
    _, isos = sortISOs(list(data[components[0]]['results'][image].keys()))
    #isos = baseisos + isos
    for component in components:
        try:
            ssimscore = [data[component]['results'][image][iso][args.metric] for iso in isos]
            plt.plot(isos, ssimscore, label=component, marker=markers[component])
            plt.title(image)
            if args.metric == 'ssim':
                plt.ylabel('SSIM score')
            else:
                plt.ylabel('MSE loss')
            plt.xlabel('ISO value')
        except KeyError as err:
            print(err)
            continue
    plt.grid()
    plt.legend()
    if not args.noshow:
        plt.show()
# TODO use json to handle nested dicts
if not args.nojson:
    with open('res.json', 'w') as f:
        json.dump(data, f)
