# -*- coding: utf-8 -*-
'''
pick validation set:
    requires a cropped dataset
    should only be done once per test_dir for consistent results
    generates validation_crops_<NUM_CROPS>_<TRAIN_DATA>_<get_fn(TEST_SET_YAML)>
    TODO copy crops to visualise ground-truth
'''
import configargparse
import os
import random
import yaml
import sys
sys.path.append('..')
from nind_denoise import nn_common
from nind_denoise import dataset_torch_3
from nind_denoise.nn_train import DEFAULT_CONFIG_FPATH
from common.libs import utilities

parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
    nn_common.COMMON_CONFIG_FPATH, DEFAULT_CONFIG_FPATH],
    config_file_parser_class=configargparse.YAMLConfigFileParser)
parser.add_argument('--train_data', nargs='*',
                    help='Draw crops randomly from this training image directory')
parser.add_argument('--num_crops', type=int, default=30)
parser.add_argument('--test_reserve', nargs='*', required=True, help='Space separated list of image sets to be reserved for testing, or yaml file path containing a list. Set to "0" to use all available data.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing validation set?')
args,_ = parser.parse_known_args()

test_reserve_str = utilities.get_leaf(args.test_reserve[0]) # assumes yaml file
assert test_reserve_str.endswith('.yaml')
args.test_reserve = nn_common.get_test_reserve_list(args.test_reserve)
# with open(args.test_reserve_yaml, 'r') as fp:
#     test_sets = yaml.safe_load(fp)
    
# find crops
crops_paths = []  # list of baseline-noisy tuples
train_data_dnames = []
for train_data_dpath in args.train_data:
    train_data_dnames.append(os.path.basename(os.path.relpath(train_data_dpath)))
    for set_name in os.listdir(train_data_dpath):
        if set_name not in args.test_reserve:
            continue
        set_dpath = os.path.join(train_data_dpath, set_name)
        ISOs = os.listdir(set_dpath)
        base_isos, isos = dataset_torch_3.sortISOs(ISOs)
        for base_iso in base_isos:
            base_iso_dpath = os.path.join(set_dpath, base_iso)
            crops_filenames = os.listdir(base_iso_dpath)
            for noisy_iso in isos:
                noisy_iso_dpath = os.path.join(set_dpath, noisy_iso)
                for crop_fn in crops_filenames:
                    crops_paths.append([os.path.join(base_iso_dpath, crop_fn),
                                        os.path.join(noisy_iso_dpath, crop_fn.replace(base_iso, noisy_iso))])

res_fpath = os.path.join('configs', f'validation_set_{args.num_crops}_{"+".join(train_data_dnames)}_{os.path.basename(os.path.relpath(test_reserve_str))}')

if os.path.isfile(res_fpath) and not args.overwrite:
    sys.exit(f'{res_fpath} exists and args.overwrite is not set')


# randomly select crops
chosen_crops = random.sample(crops_paths, args.num_crops)

# test that crops exist
for acrop in chosen_crops:
    assert os.path.isfile(acrop[0]), acrop
    assert os.path.isfile(acrop[1]), acrop

# write to yaml
with open(res_fpath, 'w') as fp:
    yaml.dump(chosen_crops, fp)

print(f'{chosen_crops=} written to {res_fpath=}')