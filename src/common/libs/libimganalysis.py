'''
Image analysis on file paths
'''

import piqa
import subprocess
try:
    import piexif
except ModuleNotFoundError:
    print('filter_dataset_by_iso.py: warning: piexif library not found, using exiftool instead')
import sys
sys.path.append('..')
from common.libs import pt_helpers

def piqa_msssim(img1path: str, img2path: str):
    img1 = pt_helpers.fpath_to_tensor(img1path, batch=True)
    img2 = pt_helpers.fpath_to_tensor(img2path, batch=True)
    return piqa.MS_SSIM()(img1, img2).item()

def get_iso(fpath):
    def piexif_get_iso(fpath):
        '''
        supports jpeg and maybe tiff. should be slightly faster than calling exiftool.
        '''
        try:
            exifdata = piexif.load(fpath)['Exif']
        except Exception as e:
            print(f'piexif_get_iso: {e} on {fpath}; reverting to exiftool_get_iso')
            return exiftool_get_iso(fpath)
        if 34855 in exifdata:
            isoval = exifdata[34855]
            if not isinstance(isoval, int):
                print(f'piexif_get_iso: invalid non-int format for {fpath} ({isoval}), skipping.')
                isoval = None
            return isoval
    def exiftool_get_iso(fpath):
        cmd = 'exiftool', '-S', '-ISO', fpath
        try:
            res = subprocess.run(cmd, text=True, capture_output=True).stdout
        except FileNotFoundError:
            exit('exiftool_get_iso: exiftool binary not present') 
        if res == '':
            return None
        else:
            try:
                return int(res.split(': ')[-1])
            except ValueError as e:
                print(f'exiftool_get_iso: got {e} on {fpath}, skipping.')
    ext = fpath[-4:].lower()
    isoval = None
    if ext.endswith('jpg') or ext.endswith('jpeg'):
        isoval = piexif_get_iso(fpath)
    else:
        isoval = exiftool_get_iso(fpath)
    assert isoval is None or isinstance(isoval, int), f'{fpath=}, {isoval=}'
    return isoval