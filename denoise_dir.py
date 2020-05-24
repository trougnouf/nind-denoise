# run this to test the model: denoise a directory (datasets/test/ds_fs) and computes the SSIM score for each (using the lowest-ISO ground-truth located in the same directory), store results in results/test/<model_name>/res.txt

import argparse
import os
import time
import sys
import torch
import torchvision
from PIL import Image
from loss import gen_score
import subprocess
from nn_common import Model, default_values

# eg python denoise_dir.py --model_subdir ...
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_dir', default='datasets/test/ds_fs', type=str, help='directory of test dataset (or any directory containing images to be denoised), must end with [CROPSIZE]_[USEFULCROPSIZE]')
    parser.add_argument('--network', type=str, default=default_values['g_network'], help='Generator network (default: %s)'%default_values['g_network'])
    parser.add_argument('--model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
    parser.add_argument('--model_parameters', default="", type=str, help='Model parameters with format "parameter1=value1,parameter2=value2"')
    parser.add_argument('--result_dir', default='results/test', type=str, help='directory where results are saved')
    parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--no_scoring', action='store_true', help='Generate SSIM score and MSE loss unless this is set')
    parser.add_argument('--cs', type=str, default='128') # TODO compute acceptable values
    parser.add_argument('--ucs', type=str, default='112')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    assert args.model_path is not None
    model_path = Model.complete_path(args.model_path, keyword='generator')

    sets_to_denoise = os.listdir(args.noisy_dir)
    denoised_save_dir=os.path.join(args.result_dir, model_path.split('/')[-2])
    os.makedirs(denoised_save_dir, exist_ok=True)
    for aset in sets_to_denoise:
        aset_indir = os.path.join(args.noisy_dir, aset)
        for animg in os.listdir(aset_indir):
            inimg_path = os.path.join(aset_indir, animg)
            outimg_path = os.path.join(denoised_save_dir, animg)
            cmd = ['python', 'denoise_image.py', '-i', inimg_path, '-o', outimg_path, '--model_path', model_path, '--network',args.network, '--model_parameters', args.model_parameters, '--ucs', args.ucs, '--cs', args.cs]
            subprocess.call(cmd)
    if not args.no_scoring:
        gen_score(denoised_save_dir, args.noisy_dir)
