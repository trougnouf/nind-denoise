'''
List a datasets' crops under a csv file and list all ms-ssim values
Useful to train above a given quality threshold
'''
import configargparse
import os
import sys
sys.path.append('..')
from nind_denoise import nn_common
from common.libs import utilities
from PIL import Image
import subprocess
import shutil
from tqdm import tqdm
from common.libs import utilities
from common.libs import libimganalysis

DATA_DPATH = os.path.join('..', '..', 'datasets', 'FeaturedPictures')
MAXISO = 200



if __name__ == '__main__':
    
    # Training settings
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
        nn_common.COMMON_CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    #parser.add('-c', '--config', is_config_file=True, help='(yaml) config file path')
    parser.add_argument('--data_dpath', default=DATA_DPATH, help="Directory where the data to filter is located")
    parser.add_argument('--out_dpath', help="Filtered data output directory (default: data_dpath/../filtered/data_dname")
    parser.add_argument('--maxISO', type=int, default=MAXISO, help='Maximum ISO value')
    args, _ = parser.parse_known_args()
    
    if args.out_dpath is None:
        args.out_dpath = os.path.join(args.data_dpath, '..', 'filtered', f'ISO{args.maxISO}', utilities.get_leaf(args.data_dpath))
    os.makedirs(args.out_dpath, exist_ok=True)
    for fn in tqdm(os.listdir(args.data_dpath)):
        
        infpath = os.path.join(args.data_dpath, fn)
        isoval = libimganalysis.get_iso(infpath)
        if isoval is not None and isoval <= MAXISO:
            outfpath = os.path.join(args.out_dpath, fn)
            if not os.path.exists(outfpath):
                utilities.cp(infpath, outfpath)
                #shutil.copy2(infpath, outfpath)
