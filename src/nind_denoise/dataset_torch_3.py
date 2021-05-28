'''
NIND dataset handler for pytorch. Loads the pre-cropped dataset, returns clean,
noisy crops where noise value is randomized (unless specified in yval).
Supports on-the-fly compression (compressionmin, compressionmax), artificial
noise (sigmamin, sigmamax), and test_reserve (with exact_reserve or keyword search)
Images have 3 sizes:
    - cs or crop size:
        output crop size (typically but not necessarily same as input)
    - ucs or useful crop size:
        minimum valid input crop size; usually this is the stride used to create input crops.
        sometimes used as the central crop for evaluation
See also: DenoisingDataset class
'''
# TODO: add sharpening as optional data augmentation
import os
#from PIL import Image, ImageOps
import cv2
import torchvision
#from random import randint, uniform, choice
import random
from io import BytesIO
import torch
import unittest
import numpy as np
import yaml
import csv
from typing import Optional, List
from PIL import Image # only used for experimental JPEG compression artifact removal
import sys
sys.path.append('..')
from common.libs import np_imgops
from common.libs import pt_ops
from common.libs import pt_helpers
from common.libs import libimganalysis
from common.libs import utilities

FS_DS_DPATH = os.path.join('..', '..', 'datasets', 'NIND')
CROPPED_DS_DPATH = os.path.join('..', '..', 'datasets', 'cropped')

def sortISOs(rawISOs: List[str]) -> tuple:
    '''
    Sort ISO values (eg ISO200, ISO6400, ...), handles ISOH1, ISOH2, ..., ISOHn as last,
    handles ISO200-n, ISO6400-n, ... as usable duplicates

    ISO directories can be ISO<NUM>[-REPEATNUM] or ISOH<NUM>[-REPEATNSUM]
    where ISO<lowest> (and possibly any -REPEATNUM, for example ISO200-1 and
    ISO200-2) is taken as the base ISO. Other naming conventions will also work
    so long as the base iso is first alphabetically. GT* and ISO* can handle
    base ISOs
    '''
    isos = []
    bisos = []
    if any([iso[:3] != 'ISO' for iso in rawISOs]):
        for iso in rawISOs:
            if 'GT' in iso:
                bisos.append(iso)
            else:
                isos.append(iso)
        isos=sorted(isos)
        if len(bisos)==0:
            bisos.append(isos.pop(0))
        return bisos, isos
    hisos = []
    dupisos = {}
    for iso in rawISOs:
        if 'H' in iso:
            hisos.append(iso)
        else:
            if '-' in iso:
                isoval, _, repid = iso[3:].partition('-')
                isos.append(int(isoval))
                if isoval in dupisos:
                    dupisos[isoval].append(repid)
                else:
                    dupisos[isoval] = [repid]
            else:
                isos.append(int(iso[3:]))
    bisos,*isos = sorted(isos)
    bisos = [bisos]
    # add duplicates
    while(bisos[0]==isos[0]):
        bisos.append(str(isos.pop(0))+'-'+dupisos[str(bisos[0])].pop())
    for dupiso in dupisos.keys():
        for repid in dupisos[dupiso]:
            isos[isos.index(int(dupiso))] = dupiso+'-'+repid
    hisos = sorted(hisos)
    isos = ['ISO'+str(iso) for iso in isos]
    bisos = ['ISO'+str(iso) for iso in bisos]
    isos.extend(hisos)
    return bisos, isos

def get_baseline_fpath(dpath: str) -> str:
    '''
    input: directory containing whole image sets (eg: ../../datasets/NIND/banana which contains NIND_banana_ISO<isovalue>.png files)
    output: a baseline image (eg: NIND_banana_ISO200.png)
    '''
    iso_fn_dict = {fn.split('_')[-1].split('.')[0]: fn for fn in os.listdir(dpath)}
    bisos, _ = sortISOs(iso_fn_dict.keys())
    return os.path.join(dpath, iso_fn_dict[bisos[0]])

class DenoisingDataset(torch.utils.data.Dataset):
    '''
    DenoisingDataset: pytorch dataset of clean/noisy images s.a. the Natural Image Noisy Dataset
    (pre-cropped)
    
    datadirs: str:
        expects path to directory of crops with the following structure:
        <datadirs>/<set>/ISO<ISOvalue>/*_ucs.ext
        where datadirs should be named "<DSNAME>_<CS>_<UCS>"
    test_reserve: list:
        list of sets reserved for testing
    exact_reserve: boolean:
        true: only reserve sets named in the test_reserve.
        false: reserve all sets which contain a string present in the test_reserve
    cs: output crop size (padded or randomly cropped if necessary). if None, autodetected from ds name
    min_crop_size: Optional[int]: minimum input crop size; dataset is checked if set
    min/max_exp_mult: image values are multiplied by these factors (data augmentation)
    
    rarely used / experimental parameters:
    
    yval:
        use to train for a single ISO value instead of a generalizing network (None).
        Can also be set to 'x' to train with base ISOs only
    compressionmin, compressionmax:
        apply JPEG compresison to the noisy images
    sigmamin, sigmamax:
        add artificial noise to noisy images if sigma > 0
    
    '''
    def __init__(self, datadirs: List[str], yval: Optional[str]=None, compressionmin: int = 100,
                 compressionmax: int = 100, sigmamin: int = 0, sigmamax: int = 0,
                 test_reserve: list = [], min_crop_size: Optional[int] = None,
                 exact_reserve: bool = False, cs=None, exp_mult_min=1, exp_mult_max=1):
        def keep_only_isoval_from_list(isos,keepval):
            keptisos = []
            for iso in isos:
                if iso.endswith(keepval) or iso.endswith(keepval+'-'):
                    keptisos.append(iso)
            return keptisos
        def is_reserved(aset):
            if exact_reserve:
                if test_reserve and aset in test_reserve:
                    return True
            elif test_reserve:
                for skip_string in test_reserve:
                    if skip_string in aset:
                        return True
            return False
        super(DenoisingDataset, self).__init__()
        self.totensor = torchvision.transforms.ToTensor()
        # each dataset element is ["<DATADIR>/<SETNAME>/ISOBASE/<DSNAME>_<SETNAME>_ISOBASE_<XNUM>_<YNUM>_<UCS>.EXT", [<ISOVAL1>,...,<ISOVALN>]]
        self.dataset = []
        # determine cs, ucs if not provided
        check_dataset = min_crop_size is not None
        self.min_crop_size = min_crop_size
        if cs is not None:
            self.cs = cs
        else:
            self.cs, min_crop_size = [int(i) for i in datadirs[0].split('_')[-2:]]
            if self.min_crop_size is None:
                self.min_crop_size = min_crop_size
        assert self.cs is not None
        self.compressionmin, self.compressionmax = compressionmin, compressionmax
        self.sigmamin, self.sigmamax = sigmamin, sigmamax
        self.exp_mult_min, self.exp_mult_max = exp_mult_min, exp_mult_max
        # scan given dataset directory
        for datadir in datadirs:
            for aset in os.listdir(datadir):
                if is_reserved(aset):
                    print('Skipped '+aset+' (test reserve)')
                    continue
                bisos, isos = sortISOs(os.listdir(os.path.join(datadir,aset)))
                if yval is not None:
                    if yval == 'x':
                        bisos = isos = bisos[0:1]
                    # TODO add maximum ISO value
                    else:
                        isos = keep_only_isoval_from_list(isos, yval)
                        if len(isos) == 0:
                            print('Skipped '+aset+' ('+yval+' not found)')
                            continue
                # remove the following (sizecheck) if problematic
                # check for min size
                
                for animg in os.listdir(os.path.join(datadir, aset, isos[0])):
                    imgpath = os.path.join(datadir, aset, isos[0], animg)
                    if check_dataset: 
                        imgdims = cv2.imread(imgpath, flags=cv2.IMREAD_COLOR+cv2.IMREAD_ANYDEPTH).shape[:2]
                        if any(d < self.min_crop_size for d in imgdims):
                            print(f'DenoisingDataset: skipping {imgpath} because {imgdims} < {self.min_crop_size}')
                            continue
                    self.dataset.append([os.path.join(datadir,aset,'ISOBASE',animg).replace(isos[0]+'_','ISOBASE_'), bisos,isos])
                print('Added '+aset+str(bisos)+str(isos)+' to the dataset')
        self.dsname = '+'.join([utilities.get_leaf(path) for path in datadirs])

    def get_x_y_paths(self, index):
        img = self.dataset[index]
        xchoice = random.choice(img[1])
        xpath = os.path.join(img[0].replace('ISOBASE_',xchoice+'_').replace('/ISOBASE/','/'+xchoice+'/'))
        ychoice = random.choice(img[2])
        ypath = os.path.join(img[0].replace('ISOBASE_',ychoice+'_').replace('/ISOBASE/','/'+ychoice+'/'))
        return xpath, ypath
    
    def get_all_crop_pairs_of_paths(self) -> tuple:
        '''
        Returns file path of all crop pairs (gt, noisy)
        Useful for debugging, eg to compute loss between all pairs
        '''
        for el in self.dataset:
            for biso in el[1]:
                for noisy_iso in el[2]:
                    gt_crop_path = os.path.join(el[0].replace('ISOBASE_',biso+'_').replace('/ISOBASE/','/'+biso+'/'))
                    noisy_crop_path = os.path.join(el[0].replace('ISOBASE_',noisy_iso+'_').replace('/ISOBASE/','/'+noisy_iso+'/'))
                    yield gt_crop_path, noisy_crop_path
                    
    def list_content_quality(self, export=False):
        '''
        Can be useful to check the dataset quality and find bad crops
        output to ../../datasets/cropped/msssim.csv
        '''
        scores = list()
        for xpath, ypath in self.get_all_crop_pairs_of_paths():
            score = (xpath, ypath, libimganalysis.piqa_msssim(xpath, ypath))
            scores.append(score)
            print(score)
        if export:
            outpath = os.path.join('datasets', self.dsname+'-msssim.csv')
            utilities.list_of_tuples_to_csv(scores, ('xpath', 'ypath', 'score'), outpath)
            print(f'Quality check exported to {outpath}')
        return scores
        
    def crop_and_pad_from_paths(self, xpath, ypath):
        '''
        filenames are expected to end with "_<UCS>.<ext>
        '''
        ximg = np_imgops.img_path_to_np_flt(xpath)
        yimg = np_imgops.img_path_to_np_flt(ypath)
        assert ximg.shape == yimg.shape, 'Error: crops do not match: '+xpath+', '+ypath
        if any(d < self.cs for d in ximg.shape[1:]):
            ximg, yimg = np_imgops.np_pad_img_pair(ximg, yimg. self.cs)
        if any(d > self.cs for d in ximg.shape[1:]):
            ximg, yimg = np_imgops.np_crop_img_pair(ximg, yimg, self.cs, np_imgops.CropMethod.RAND)
        assert all(d == self.cs for d in ximg.shape[1:]), f'{ximg.shape=}, {self.cs=}'
        return (ximg, yimg)

    def __getitem__(self, reqindex):
        xpath, ypath = self.get_x_y_paths(reqindex)
        ximg, yimg = self.crop_and_pad_from_paths(xpath, ypath)
        # data augmentation
        nrot = random.randint(0, 3)
        ximg = np.rot90(ximg, nrot, (1, 2))
        yimg = np.rot90(yimg, nrot, (1, 2))
        if random.getrandbits(1):
            ximg = np.flip(ximg, 1)
            yimg = np.flip(yimg, 1)
        if random.getrandbits(1):
            ximg = np.flip(ximg, 2)
            yimg = np.flip(yimg, 2)
        if getattr(self, 'compressionmin', 100) < 100:
            quality = random.randint(self.compressionmin, self.compressionmax)
            imbuffer = BytesIO()
            yimg.save(imbuffer, 'JPEG', quality=quality)
            yimg = Image.open(imbuffer)
        
        # return a tensor
        
        ximg, yimg = torch.tensor(ximg.copy()), torch.tensor(yimg.copy())
        if getattr(self, 'sigmamax', 0) > 0:
            noise = torch.randn(yimg.shape).mul_(random.uniform(self.sigmamin, self.sigmamax)/255)
            yimg = torch.abs(yimg+noise)
        if self.exp_mult_min < 1 or self.exp_mult_max > 1:
            exp_mult = random.uniform(self.exp_mult_min, min(self.exp_mult_max, 1/ximg.max()))
            ximg = (ximg * exp_mult)
            yimg = (yimg * exp_mult).clip(0, 1)
        return ximg, yimg
    
    def __len__(self):
        return len(self.dataset)

class PickyDenoisingDatasetFromList(DenoisingDataset):
    '''
    Similar to DenoisingDataset, but takes a csv file containing paths to ground-truth - noisy crops
    and their ms-ssim quality (gt-noisy-msssim tuple), instead of relying on the directory structure.
    A minimum MS-SSIM threshold can be set.
    The csv file can be generated by DenoisingDataset.list_content_quality and
    tools/make_dataset_crops_list.py 
    '''
    def __init__(self, csv_fpath, min_quality=0, exp_mult_min=1, exp_mult_max=1):
        self.dataset = []
        with open(csv_fpath, 'r') as fp:
            for acrop in csv.DictReader(fp):
                if acrop['score'] > min_quality:
                    self.dataset.append({'xpath': acrop['xpath'], 'ypath': acrop['ypath']})
        self.exp_mult_min = exp_mult_min
        self.exp_mult_max = exp_mult_max
    def get_x_y_paths(self, i):
        return self.dataset['xpath'], self.dataset['ypath']

class LazyNoiseDataset(DenoisingDataset):
    '''
    DenoisingDataset where the noisy value is returned as both x and y.
    Lazy implementation: fetches both gt/noisy values and discard gt
    '''
    def __init__(self, **vars):
        super().__init__(**vars)

    def __getitem__(self, reqindex):
        _, noisy = super().__getitem__(reqindex)
        return noisy, noisy


class TestDenoiseDataset(torch.utils.data.Dataset):
    '''
    This dataset loader returns full size images for testing.
    
    data_dpath:
        expects a directory structured as follow:
            <data_dpath>/<scene*>/
            <data_dpath>/<scene*>/<noisy*>.<ext>
            <data_dpath>/<scene*>/<noisy*>.<ext>
            <data_dpath>/<scene*>/<noisy*>.<ext>
            <data_dpath>/<scene*>/gt/<ground-truth>.<ext>
        Alternatively the original dataset directory (as downloaded by dl_ds_1.py) can be provided,
        base ISO will then be autodetected.
    
    sets: list of sets that will be returned (eg: same as DenoisingDataset's test_reserve)
    
#     If the val init parameter is set to True, then only one noisy image is returned per scene.
#     Different resolutions -> batch size must be 1
    '''
    def __init__(self, data_dpath, val=False, sets=[]):
        super().__init__()
        scenes = os.listdir(data_dpath)
        val_i = 0
        
        self.ds = []
        for ascene in scenes:
            ascene_dpath = os.path.join(data_dpath, ascene)
            if len(sets) > 0 and ascene not in sets:
                continue
            if os.path.isdir(os.path.join(ascene_dpath, 'gt')):
                # gt directory provided:
                gt_fpath = os.path.join(ascene_dpath, 'gt', os.listdir(os.path.join(ascene_dpath, 'gt'))[0])
                noisy_fpaths = [os.path.join(ascene_dpath, fn) for fn in os.listdir(ascene_dpath)]
                noisy_fpaths.remove(os.path.join(ascene_dpath, 'gt'))
            else:
                # autodetect base ISO:
                ISOvals = [fn.split('_')[-1].split('.')[0] for fn in os.listdir(ascene_dpath)]
                gtval, noisyvals = sortISOs(ISOvals)
                noisy_fpaths = []
                gt_fpath = None
                for fn in os.listdir(ascene_dpath):
                    if gtval[0]+'.' in fn:
                        gt_fpath = os.path.join(ascene_dpath, fn)
                    else:  # FIXME? This will include any alternative base-ISOs with the noisy images
                        noisy_fpaths.append(os.path.join(ascene_dpath, fn))
#             if val:
#                 try:
#                     noisy_fpaths = [noisy_fpaths[val_i]]
#                 except IndexError:
#                     val_i = 0
#                     noisy_fpaths = [noisy_fpaths[val_i]]
#                 val_i += 1
            self.ds.append({'gt': gt_fpath, 'noisy': noisy_fpaths})
    def get_imgs(self):
        for ascene in self.ds:
            gt = pt_helpers.fpath_to_tensor(ascene['gt'])
            for noisy_fpath in ascene['noisy']:
                noisy = pt_helpers.fpath_to_tensor(noisy_fpath)
                yield gt, noisy

    def __getitem__(self, index):
        # not very clever but no high perf needed here. ideally use get_imgs without pt dataloader.
        i = 0
        for ascene in self.ds:
            for noisy_fpath in ascene['noisy']:
                if i == index:
                    gt = pt_helpers.fpath_to_tensor(ascene['gt'])
                    noisy = pt_helpers.fpath_to_tensor(noisy_fpath)
                    return pt_ops.crop_to_multiple(gt, 64), pt_ops.crop_to_multiple(noisy, 64)
                i += 1

    def __len__(self):
        length = 0
        for ascene in self.ds:
            length += len(ascene['noisy'])
        return length

class ValidationDataset(torch.utils.data.Dataset):
    '''
    val_tuples is a list of clean-noisy tuples* representing image paths (ideally crops)
    val_tuples can be the path to a yaml file containing the aforementioned list
    
    This small subset of the test set is used during training (to evaluate performance and update lr)
    
    * actually lists because tuples don't exist in yaml
    '''
    def __init__(self, val_tuples, device, cs):
        if isinstance(val_tuples, str):
            # assuming yaml
            with open(val_tuples, 'r') as fp:
                self.val_tuples = yaml.safe_load(fp)
        else:
            self.val_tuples = val_tuples
        self.device = device
        self.cs = cs
    def __len__(self):
        return len(self.val_tuples)
    def __getitem__(self, i):
        #np_res = self.get_and_pad_paths(self.val_tuples[i][0], self.val_tuples[i][1])
        ximg = np_imgops.img_path_to_np_flt(self.val_tuples[i][0])
        yimg = np_imgops.img_path_to_np_flt(self.val_tuples[i][1])
        ximg, yimg = np_imgops.np_crop_img_pair(ximg, yimg, self.cs, np_imgops.CropMethod.CENTER)
        return torch.tensor(ximg, device=self.device), torch.tensor(yimg, device=self.device)


# TODO save img fun

class Test_DS(unittest.TestCase):
    def setUp(self):

        for fn in os.listdir('unittest_resources'):
            if '.png' in fn:
                self.img16bpath = os.path.join('unittest_resources', fn)
            elif '.jpg' in fn:
                self.img8bpath = os.path.join('unittest_resources', fn)
    def test_ds_exists(self):
        if (os.path.isdir(CROPPED_DS_DPATH) and len(os.listdir(CROPPED_DS_DPATH)) > 0):
            self.dsdir = os.path.join(CROPPED_DS_DPATH, os.listdir(CROPPED_DS_DPATH)[0])
        elif (os.path.isdir(CROPPED_DS_DPATH)
              and sum(['NIND' in os.listdir(os.path.isdir(CROPPED_DS_DPATH))]) > 0):
            for dsdir in os.listdir(os.path.isdir(CROPPED_DS_DPATH)):
                if 'NIND' in dsdir:
                    self.dsdir = os.path.join(CROPPED_DS_DPATH, dsdir)
                    break
        self.assertIsNotNone(self.dsdir,
                             msg="""Test_DS.setUp: cropped dataset directory not found (eg: in datasets/train/).
                             First run dl_ds_1.py and crop_ds.py.""")
    def test_rotate(self):
        from PIL import Image
        pilimg = Image.open(self.img8bpath)
        rotpilimg = pilimg.rotate(90)
        npfromrotpil = np.transpose(np.array(rotpilimg), (2,0,1)).astype(np.single)/255
        npimg = np_imgops.img_path_to_np_flt(self.img8bpath)
        rotnpimg = np.rot90(npimg, 1, (1,2))
        # print('pil')
        # print(npfromrotpil.shape)
        # print(npfromrotpil)
        # print('np')
        # print(rotnpimg.shape)
        # print(rotnpimg)
        self.assertTrue(np.array_equal(npfromrotpil, rotnpimg))



if __name__ == '__main__':
    print('test dataset:')
    tds = TestDenoiseDataset('../../datasets/test/NIND', val=False)
    for gt, noisy in tds.get_imgs():
        print('{}, {}'.format(gt.shape, noisy.shape))
    vds = TestDenoiseDataset('../../datasets/test/NIND', val=True)
    print('val dataset:')
    for gt, noisy in vds.get_imgs():
        print('{}, {}'.format(gt.shape, noisy.shape))