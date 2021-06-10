'''
List a datasets' crops under a csv file and list all ms-ssim values
Useful to train above a given quality threshold
'''
import configargparse
import os
import sys
sys.path.append('..')
from nind_denoise import dataset_torch_3
from nind_denoise import nn_common
from nind_denoise import nn_train
from common.libs import utilities
            
if __name__ == '__main__':
    
    # Training settings
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
        nn_common.COMMON_CONFIG_FPATH, nn_train.DEFAULT_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='(yaml) config file path')
    parser.add_argument('--test_reserve', nargs='*', required=True, help='Space separated list of image sets to be reserved for testing, or yaml file path containing a list. Set to "0" to use all available data.')
    parser.add_argument('--train_data', nargs='*', help="(space-separated) Path(s) to the pre-cropped training data")
    parser.add_argument('--cs', '--crop_size', type=int, help='Crop size fed to NN. default: no additional cropping')
    parser.add_argument('--min_crop_size', type=int, help='Minimum crop size. Dataset will be checked if this value is set.')
    parser.add_argument('--loss_cs', '--loss_crop_size', type=int, help='Center crop size used in loss function. default: use stride size from dataset directory name')
    parser.add_argument('--debug_options', '--debug', nargs='*', default=[], help=f"(space-separated) Debug options (available: {nn_common.DebugOptions})")
    args, _ = parser.parse_known_args()
    debug_options = [nn_common.DebugOptions(opt) for opt in args.debug_options]
    # Train data
    if (args.min_crop_size is None or args.min_crop_size == 0) and nn_common.DebugOptions.CHECK_DATASET in debug_options:
        args.min_crop_size = args.cs
    DDataset = dataset_torch_3.DenoisingDataset(args.train_data, test_reserve=args.test_reserve,
                                                cs=args.cs, min_crop_size=args.min_crop_size)
    xpaths_ypaths_scores = DDataset.list_content_quality()
    
    outpath = os.path.join('datasets', DDataset.dsname+'-msssim.csv')
    os.makedirs('datasets', exist_ok=True)
    scores = DDataset.list_content_quality()
    utilities.list_of_tuples_to_csv(scores, ('xpath', 'ypath', 'score'), outpath)
    print(f'Quality check exported to {outpath}')