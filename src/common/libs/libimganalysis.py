'''
Image analysis on file paths
'''

import piqa
import sys
sys.path.append('..')
from common.libs import pt_helpers

def piqa_msssim(img1path: str, img2path: str):
    img1 = pt_helpers.fpath_to_tensor(img1path, batch=True)
    img2 = pt_helpers.fpath_to_tensor(img2path, batch=True)
    return piqa.MS_SSIM()(img1, img2).item()