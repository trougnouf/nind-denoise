# NIND dataset handler for pytorch. Loads the pre-cropped dataset, returns clean, noisy crops where noise value is randomized (unless specified in yval). Supports on-the-fly compression (compressionmin, compressionmax), artificial noise (sigmamin, sigmamax), and test_reserve (with exact_reserve or keyword search)

import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision
from random import randint, uniform, choice
from math import floor
from io import BytesIO
import torch

# Sort ISO values (eg ISO200, ISO6400, ...), handles ISOH1, ISOH2, ..., ISOHn as last, handles ISO200-n, ISO6400-n, ... as usable duplicates
def sortISOs(rawISOs):
    # ISO directories can be ISO<NUM>[-REPEATNUM] or ISOH<NUM>[-REPEATNSUM]
    # where ISO<lowest> (and possibly any -REPEATNUM, for example ISO200-1 and
    # ISO200-2) is taken as the base ISO. Other naming conventions will also work
    # so long as the base iso is first alphabetically. GT* and ISO* can handle
    # base ISOs
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

class DenoisingDataset(Dataset):
    def __init__(self, datadirs, testreserve=[], yval=None, compressionmin=100, compressionmax=100, sigmamin=0, sigmamax=0, test_reserve=[], do_sizecheck=False, exact_reserve=False):
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
        self.cs, self.ucs = [int(i) for i in datadirs[0].split('_')[-2:]]
        self.compressionmin, self.compressionmax = compressionmin, compressionmax
        self.sigmamin, self.sigmamax = sigmamin, sigmamax
        for datadir in datadirs:
            for aset in os.listdir(datadir):
                if is_reserved(aset):
                    print('Skipped '+aset+' (test reserve)')
                    continue
                bisos, isos = sortISOs(os.listdir(os.path.join(datadir,aset)))
                if yval is not None:
                    if yval == 'x':
                        bisos = isos = bisos[0:1]
                    else:
                        isos = keep_only_isoval_from_list(isos, yval)
                        if len(isos) == 0:
                            print('Skipped '+aset+' ('+yval+' not found)')
                            continue
                # check for min size
                for animg in os.listdir(os.path.join(datadir, aset, isos[0])):
                    if not do_sizecheck:
                        imgdims = [int(animg.split('_')[-1].split('.')[0])]
                    else:
                        imgdims = Image.open(os.path.join(datadir, aset, isos[0], animg)).size
                    # check for min size
                    #img4tests=Image.open(os.path.join(datadir, aset, isos[0], animg))
                    if all(d >= self.ucs for d in imgdims):
                        self.dataset.append([os.path.join(datadir,aset,'ISOBASE',animg).replace(isos[0]+'_','ISOBASE_'), bisos,isos])
                    else:   # dbg check for quick check, if nothing prints then the above test can safely be removed
                        if int(animg.split('_')[-1].split('.')[0]) >= self.ucs:
                            print('Warning: UCS FN does not match: '+animg.split('_')[-1].split('.')[0])
                    # verify that no base-ISO image exceeds CS just because
                    if any(d > self.cs for d in imgdims):
                        print("Warning: excessive crop size for "+aset)
                print('Added '+aset+str(bisos)+str(isos)+' to the dataset')

    def get_and_pad(self, index):
        img = self.dataset[index]
        xchoice = choice(img[1])
        xpath = os.path.join(img[0].replace('ISOBASE_',xchoice+'_').replace('/ISOBASE/','/'+xchoice+'/'))
        ychoice = choice(img[2])
        ypath = os.path.join(img[0].replace('ISOBASE_',ychoice+'_').replace('/ISOBASE/','/'+ychoice+'/'))
        ximg = Image.open(xpath)
        if ximg.getbands() != ('R', 'G', 'B'):
            ximg=ximg.convert('RGB')
        yimg = Image.open(ypath)
        if yimg.getbands() != ('R', 'G', 'B'):
            yimg=yimg.convert('RGB')
        if ximg.size != yimg.size:
            print('Warning: crops do not match: '+xpath+', '+ypath)
            return self.get_and_pad(index)
        if all(d == self.cs for d in ximg.size):
            return (ximg, yimg)
        xnum, ynum, ucs = [int(i) for i in img[0].rpartition('.')[0].split('_')[-3:]]
        if xnum == 0:
            # pad left
            ximg = ximg.crop((-self.cs+ximg.width, 0, ximg.width, ximg.height))
            yimg = yimg.crop((-self.cs+yimg.width, 0, yimg.width, yimg.height))
        if ynum == 0:
            # pad top
            ximg = ximg.crop((0, -self.cs+ximg.height, ximg.width, ximg.height))
            yimg = yimg.crop((0, -self.cs+yimg.height, yimg.width, yimg.height))
        if ximg.width < self.cs or ximg.height < self.cs:
            # pad right and bottom
            ximg = ximg.crop((0, 0, self.cs, self.cs))
            yimg = yimg.crop((0, 0, self.cs, self.cs))
        return (ximg, yimg)

    def __getitem__(self, reqindex):
        ximg, yimg = self.get_and_pad(reqindex)
        # data augmentation
        random_decision = randint(0, 99)
        if random_decision % 10 == 0:
            ximg = ximg.rotate(90)
            yimg = yimg.rotate(90)
        elif random_decision % 10 == 1:
            ximg = ximg.rotate(180)
            yimg = yimg.rotate(180)
        elif random_decision % 10 == 2:
            ximg = ximg.rotate(270)
            yimg = yimg.rotate(270)
        if floor(random_decision/10) == 0 or floor(random_decision/10) == 2:
            ximg = ImageOps.flip(ximg)
            yimg = ImageOps.flip(yimg)
        if floor(random_decision/10) == 1 or floor(random_decision/10) == 2:
            ximg = ImageOps.mirror(ximg)
            yimg = ImageOps.mirror(yimg)
        if self.compressionmin < 100:
            quality = randint(self.compressionmin, self.compressionmax)
            imbuffer = BytesIO()
            yimg.save(imbuffer, 'JPEG', quality=quality)
            yimg = Image.open(imbuffer)
        # return a tensor
        # PIL is H x W x C, totensor is C x H x W
        ximg, yimg = self.totensor(ximg), self.totensor(yimg)
        if self.sigmamax > 0:
            noise = torch.randn(yimg.shape).mul_(uniform(self.sigmamin, self.sigmamax)/255)
            yimg = torch.abs(yimg+noise)
        return ximg, yimg
    def __len__(self):
        return len(self.dataset)
