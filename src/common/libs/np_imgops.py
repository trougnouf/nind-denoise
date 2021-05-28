import numpy as np
import random
import unittest
from enum import Enum, auto
import cv2
import os

class CropMethod(Enum):
    RAND = auto()
    CENTER = auto()

def img_path_to_np_flt(fpath):
    '''returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, y, x
    FROM common.libimgops'''
    if not os.path.isfile(fpath):
        raise FileNotFoundError(fpath)
    try:
        rgb_img = cv2.cvtColor(cv2.imread(fpath, flags=cv2.IMREAD_COLOR+cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB).transpose(2,0,1)
    except cv2.error as e:
        print(f'img_path_to_np_flp: error {e} with {fpath}')
        breakpoint()
    if rgb_img.dtype == np.ubyte:
        return rgb_img.astype(np.single)/255
    elif rgb_img.dtype == np.ushort:
        return rgb_img.astype(np.single)/65535
    else:
        raise TypeError("img_path_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})")

def np_pad_img_pair(img1, img2, cs):
    xpad0 = max(0, (cs - img1.shape[2]) // 2)
    xpad1 = max(0, cs - img1.shape[2] - xpad0)
    ypad0 = max(0, (cs - img1.shape[1]) // 2)
    ypad1 = max(0, cs - img1.shape[1] - ypad0)
    padding = ((0, 0), (ypad0, ypad1), (xpad0, xpad1))
    return np.pad(img1, padding), np.pad(img2, padding)

def np_crop_img_pair(img1, img2, cs: int, crop_method=CropMethod.RAND):
    '''
    crop an image pair into cs
    also compatible with pytorch tensors
    '''
    if crop_method is CropMethod.RAND:
        x0 = random.randint(0, img1.shape[2]-cs)
        y0 = random.randint(0, img1.shape[1]-cs)
    elif crop_method is CropMethod.CENTER:
        x0 = (img1.shape[2]-cs)//2
        y0 = (img1.shape[1]-cs)//2
    return img1[:, y0:y0+cs, x0:x0+cs], img2[:, y0:y0+cs, x0:x0+cs]


class TestImgOps(unittest.TestCase):
    def setUp(self):
        self.imgeven1 = np.random.rand(3, 8, 8)
        self.imgeven2 = np.random.rand(3, 8, 8)
        self.imgodd1 = np.random.rand(3, 5, 5)
        self.imgodd2 = np.random.rand(3, 5, 5)
        
    def test_pad(self):
        imgeven1_padded, imgeven2_padded = np_pad_img_pair(self.imgeven1, self.imgeven2, 16)
        imgodd1_padded, imgodd2_padded = np_pad_img_pair(self.imgodd1, self.imgodd2, 16)
        self.assertTupleEqual(imgeven1_padded.shape, (3, 16, 16), imgeven1_padded.shape)
        self.assertTupleEqual(imgodd2_padded.shape, (3, 16, 16), imgodd2_padded.shape)
        self.assertEqual(imgeven1_padded[0, 4, 4], self.imgeven1[0, 0, 0])
    
    def test_crop(self):
        # random crop: check size
        imgeven1_randcropped, imgeven2_randcropped = np_crop_img_pair(self.imgeven1, self.imgeven2, 4, CropMethod.RAND)
        self.assertTupleEqual(imgeven1_randcropped.shape, (3, 4, 4), imgeven1_randcropped.shape)
        
        # center crop: check size and value
        imgeven1_centercropped, imgeven2_centercropped = np_crop_img_pair(self.imgeven1, self.imgeven2, 4, CropMethod.CENTER)
        self.assertTupleEqual(imgeven1_centercropped.shape, (3, 4, 4), imgeven1_centercropped.shape)
        # orig:    0 1 2 3 4 5 6 7
        # cropped: x x 2 3 4 5 x x
        self.assertEqual(imgeven1_centercropped[0, 0, 0], self.imgeven1[0, 2, 2],
                         f'{imgeven1_centercropped[0]=}, {self.imgeven1[0]=}')
        
        # crop w/ same size: check identity
        imgeven1_randcropped, imgeven2_randcropped = np_crop_img_pair(self.imgeven1, self.imgeven2, 8, CropMethod.CENTER)
        self.assertTrue((imgeven1_randcropped == self.imgeven1).all(), 'Crop to same size is broken')
        
        
if __name__ == '__main__':
    unittest.main()