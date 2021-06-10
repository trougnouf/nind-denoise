# run this to test the model: denoise a directory (datasets/test/ds_fs) and computes the SSIM score for each (using the lowest-ISO ground-truth located in the same directory), store results in results/test/<model_name>/res.txt

'''
egrun:
Used to denoise a directory using a given model, using denoise_image.py which crops images and reassembles them.
Default is to test the model using the test_reserve and calculate the ssim and ms-ssim losses with ground-truth.
python denoise_dir.py --device -1 --model_path /orb/benoit_phd/models/nind_denoise/2021-05-27T18:45_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--debug_options_output_val_images_--test_interval_0_--epochs_1000/generator_20.pt --network UtNet --cs 552 --ucs 540
'''

import configargparse
import os
import time
import sys
import torch
import torchvision
from PIL import Image
import loss
import subprocess
from nn_common import Model
sys.path.append('..')
from common.libs import utilities
from nind_denoise import nn_common
from nind_denoise import dataset_torch_3
from common.libs import pt_helpers
from common.libs import json_saver
from nind_denoise import denoise_image


# eg python denoise_dir.py --model_subdir ...
# python denoise_dir.py --g_network UNet --model_path ../../models/nind_denoise/2019-02-18T20\:10_run_nn.py_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--skip_sizecheck_--lr_3e-4/model_257.pth
def parse_args():
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
        nn_common.COMMON_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--noisy_dir', type=str, help='directory of test dataset (or any directory containing images to be denoised), must end with [CROPSIZE]_[USEFULCROPSIZE]')
    parser.add_argument('--g_network', '--network', type=str, help='Generator network architecture (typically UtNet or UNet)')
    parser.add_argument('--model_path', '--model_fpath', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
    parser.add_argument('--model_parameters', default="", type=str, help='Model parameters with format "parameter1=value1,parameter2=value2"')
    parser.add_argument('--result_dir', default='../../results/NIND/test', type=str, help='directory where results are saved. Can also be set to "make_subdirs" to make a denoised/<model_directory_name> subdirectory')
    parser.add_argument('--cuda_device', '--device', default=0, type=int, help='Device number (default: 0, typically 0-3)')
    parser.add_argument('--no_scoring', action='store_true', help='Generate SSIM score and MSE loss unless this is set')
    parser.add_argument('--cs', type=str)
    parser.add_argument('--ucs', type=str)
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing files')
    parser.add_argument('--whole_image', action='store_true', help='Ignore cs and ucs, denoise whole image')
    parser.add_argument('--pad', type=int, help='Padding amt per side, only used for whole image (otherwise (cs-ucs)/2')
    parser.add_argument('--max_subpixels', type=int, help='Max number of pixels, otherwise abort.')
    parser.add_argument('--test_reserve', nargs='*', help='Space separated list of image sets reserved for testing, or yaml file path containing a list. Can be used like in training in place of noisy_dir argument.')
    parser.add_argument('--orig_data', help='Location of the originally downloaded train data (before cropping); used with test_reserve')
    parser.add_argument('--models_dpath', help='Directory where all models are saved')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    assert args.model_path is not None
    denoise_image.autodetect_network_cs_ucs(args)
    model_path = Model.complete_path(args.model_path, keyword='generator', models_dpath=args.models_dpath)
    if args.noisy_dir is not None:
        sets_to_denoise = os.listdir(args.noisy_dir)
        if os.path.isfile(os.path.join(args.noisy_dir, sets_to_denoise[0])):
            sets_to_denoise = ['.']  # if we are just denoising a directory containing images
        if args.result_dir == 'make_subdirs':
            denoised_save_dir = os.path.join(args.noisy_dir, '..', 'denoised', utilities.get_file_dname(args.model_path), utilities.get_leaf(args.noisy_dir))
            os.makedirs(denoised_save_dir, exist_ok=True)
        else:
            denoised_save_dir = os.path.join(args.result_dir, model_path.split('/')[-2])
        test_set_str = utilities.get_root(args.noisy_dir)
    else:
        sets_to_denoise = nn_common.get_test_reserve_list(args.test_reserve)
        args.noisy_dir = args.orig_data
        if len(args.test_reserve) == 1 and os.path.isfile(args.test_reserve[0]):
            test_set_str = utilities.get_leaf(args.test_reserve[0])
        else:
            test_set_str = str(args.test_reserve)
        
        denoised_save_dir = os.path.join(utilities.get_root(args.model_path), 'test', utilities.get_leaf(args.model_path), test_set_str)
        
    os.makedirs(denoised_save_dir, exist_ok=True)
    losses_per_set = list()
    for aset in sets_to_denoise:
        losses_per_img = list()
        aset_indir = os.path.join(args.noisy_dir, aset)
        baseline_fpath = dataset_torch_3.get_baseline_fpath(aset_indir)
        images_fn = os.listdir(aset_indir)
        for animg in images_fn:
            inimg_path = os.path.join(aset_indir, animg)
            if baseline_fpath == inimg_path:
                continue
            outimg_path = os.path.join(denoised_save_dir, animg)
            if outimg_path.endswith('jpg'):
                outimg_path = outimg_path+'.tif'
            if not (os.path.isfile(outimg_path) and args.skip_existing):
                cmd = ['python', 'denoise_image.py', '-i', inimg_path, '-o', outimg_path,
                       '--model_path', model_path, '--network',args.g_network, '--model_parameters',
                       args.model_parameters, '--ucs', str(args.ucs), '--cs', str(args.cs)]
                if args.whole_image:
                    cmd.extend(['--whole_image', '--pad', '128'])
                if args.cuda_device is not None:
                    cmd.extend(['--cuda_device', str(args.cuda_device)])
                if args.max_subpixels is not None:
                    cmd.extend(['--max_subpixels', str(args.max_subpixels)])
                    print(cmd)
                print(' '.join(cmd))
                subprocess.call(cmd)
            cur_losses = pt_helpers.get_losses(baseline_fpath, outimg_path)
            print(f'in: {inimg_path}, out: {outimg_path}, clean: {baseline_fpath}')
            print(cur_losses)
            losses_per_img.append(cur_losses)
        losses_per_set.append(utilities.avg_listofdicts(losses_per_img))
    losses_per_set = utilities.avg_listofdicts(losses_per_set)
    
    print(losses_per_set)
    
    # adding results to trainres.json. this is useful for cleanup but can't be relied on because the
    # training process could be overwriting that file.
    try:
        epoch = int(utilities.get_leaf(args.model_path).split('_')[1].split('.')[0])
        json_res_fpath = os.path.join(utilities.get_root(args.model_path), 'trainres.json')
        jsonsaver = json_saver.JSONSaver(json_res_fpath, step_type='epoch')
        jsonsaver.add_res(step=epoch, res=losses_per_set, key_prefix='test_')
    except ValueError as e:
        print(f'Cannot determine epoch from model_path {args.model_path} ({e})')
        epoch = None
    except FileNotFoundError:
        print(f'Model results json file not found ({json_res_fpath})')
    try:
        # backup test json
        json_res_fpath = os.path.join(utilities.get_root(args.model_path), 'testres.json')
        jsonsaver = json_saver.JSONSaver(json_res_fpath, step_type='epoch')
        jsonsaver.add_res(step=epoch, res=losses_per_set, key_prefix='test_')
    except TypeError as e:
        print('something is wrong with the test jsonsaver ({e}')
        # last resort
        print(f'results will be dumped to {json_res_fpath}.')
        utilities.dict_to_json(losses_per_set, json_res_fpath)
    # obsolete
    if not args.no_scoring:
        loss.gen_score(denoised_save_dir, args.noisy_dir, device=pt_helpers.get_device(args.cuda_device))
