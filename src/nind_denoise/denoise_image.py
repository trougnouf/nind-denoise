'''
Image cropper with overlap

Denoise an image: the ultimately useful test. cs and ucs should be set
depending on the network (ie UNet can handle bigger, *128Net should have
cs=128 ucs=112).

TODO make exif and such optional
TODO 16-bit output

egrun:
    whole image (lots of memory):

    python denoise_image.py --input img.tif --model_path ../../models/nind_denoise/2021-05-23T10\:16_nn_train.py_--config_configs-train_conf_unet.yaml_--debug_options_output_val_images_keep_all_output_images_--test_interval_0_--epochs_1000_--reduce_lr_factor_0.95_--patience_3/generator_250.pt --network UNet --cs 660 --ucs 470
    cropping:
        TODO
'''


import os
import argparse
import torchvision
import torch
import math
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import configargparse
from nn_common import Model
import torch.backends.cudnn as cudnn
try:
    import piexif   # TODO make it optional
except ImportError:
    pass
import subprocess
import sys
sys.path.append('..')
from common.libs import pt_helpers
from common.libs import utilities
from nind_denoise import nn_common
default_cs = 128
default_ucs = 112
default_cs_unet = 256
default_ucs_unet = 192

# TODO handle CPU
parser = configargparse.ArgumentParser(description=__doc__, default_config_files=[
    nn_common.COMMON_CONFIG_FPATH],
    config_file_parser_class=configargparse.YAMLConfigFileParser)
parser.add_argument('--cs', type=int, help='Tile size (model was probably trained with 128, different values will work with unknown results)')
parser.add_argument('--ucs', type=int, help='Useful tile size (should be <=.75*cs for U-Net, a smaller value may result in less grid artifacts but costs computation time')
parser.add_argument('-ol', '--overlap', default=6, type=int, help='Merge crops with this much overlap (Reduces grid artifacts, may reduce sharpness between crops, costs computation time)')
parser.add_argument('-i', '--input', default='in.jpg', type=str, help='Input image file')
parser.add_argument('-o', '--output', type=str, help='Output file with extension (default: model_dpath/test/denoised_images/fn.tif)')
parser.add_argument('-b', '--batch_size', type=int, default=1)  # TODO >1 is broken
parser.add_argument('--debug', action='store_true', help='Debug (store all intermediate crops in ./dbg, display useful messages)')
parser.add_argument('--cuda_device', '--device', default=0, type=int, help='Device number (default: 0, typically 0-3]], -1 for CPU)')
parser.add_argument('--exif_method', default='piexif', type=str, help='How is exif data copied over? (piexif, exiftool, noexif)')
parser.add_argument('--g_network', '--network', '--arch', type=str, help='Generator network (default: %s)')
parser.add_argument('--model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
parser.add_argument('--model_parameters', type=str, help='Model parameters with format "parameter1=value1,parameter2=value2"')
parser.add_argument('--max_subpixels', type=int, help='Max. number of sub-pixels, abort if exceeded.')
parser.add_argument('--whole_image', action='store_true', help='Ignore cs and ucs, denoise whole image')
parser.add_argument('--pad', type=int, help='Padding amt per side, only used for whole image (otherwise (cs-ucs)/2')
parser.add_argument('--models_dpath', help='Directory where all models are saved (used when a model name is provided as model_path)')

args, _ = parser.parse_known_args()

def make_output_fpath(input_fpath, model_fpath):
    model_dpath = utilities.get_root(model_fpath)
    model_fn = utilities.get_leaf(model_fpath)
    img_fn = utilities.get_leaf(input_fpath)
    os.makedirs(os.path.join(model_dpath, 'test', 'denoised_images'), exist_ok=True)
    return os.path.join(model_dpath, 'test', 'denoised_images', f'{img_fn}_{model_fn}.tif')

assert args.model_path is not None
if args.cs:
    cs = args.cs
elif 'UNet' in args.model_path or '2019-02-18' in args.model_path:
    cs = default_cs_unet
else:
    cs = default_cs
if args.ucs:
    ucs = args.ucs
elif 'UNet' in args.model_path or '2019-02-18' in args.model_path:
    ucs = default_ucs_unet
else:
    ucs = default_ucs

if args.cuda_device >= 0:
    torch.cuda.set_device(args.cuda_device)
    cudnn.benchmark = True
    torch.cuda.manual_seed(123)
torch.manual_seed(123)
device = pt_helpers.get_device(args.cuda_device)
if args.output is None:
    args.output = make_output_fpath(args.input, args.model_path)


class OneImageDS(Dataset):
    def __init__(self, inimg, cs, ucs, ol, whole_image=False, pad=None):
        self.inimg = Image.open(inimg)
        self.width, self.height = self.inimg.size
        self.totensor = torchvision.transforms.ToTensor()
        if whole_image:
            assert pad is not None
            self.pad = pad
            self.whole_image = True
            self.size = 1
        else:
            self.whole_image = False
            self.cs, self.ucs, self.ol = cs, ucs, ol    # crop size, useful crop size, overlap
            self.iperhl = math.ceil((self.width - self.ucs) / (self.ucs - self.ol)) # i_per_hline, or crops per line
            self.pad = int((self.cs - self.ucs) / 2)
            ipervl = math.ceil((self.height - self.ucs) / (self.ucs - self.ol))
            self.size = (self.iperhl+1) * (ipervl+1)
    def __getitem__(self, i):
        if self.whole_image:
            xi = yi = x0 = y0 = 0
            x1, y1 = self.width, self.height
            ret = Image.new('RGB', (x1+self.pad*2, y1+self.pad*2))
            x0pad = x1pad = y0pad = y1pad = self.pad
            crop = self.inimg
            ret.paste(crop, (x0pad, y0pad, self.width+x1pad, self.height+y1pad))
            # mirror stuff:
            # pil box argument is (left, upper, right, lower)-tuple
            if x0pad > 0:
                left_mirrored_dat = ImageOps.mirror(self.inimg.crop((0, 0, self.pad, self.height)))
                ret.paste(left_mirrored_dat, (0, self.pad, self.pad, self.height+self.pad))
            if x1pad > 0:
                right_mirrored_dat = ImageOps.mirror(self.inimg.crop((self.width-self.pad, 0, self.width, self.height)))
                ret.paste(right_mirrored_dat, (self.width+self.pad, self.pad, self.width+2*self.pad, self.height+self.pad))
            if y0pad > 0:
                top_mirrored_dat = ImageOps.flip(self.inimg.crop((0, 0, self.width, self.pad)))
                ret.paste(top_mirrored_dat, (self.pad, 0, self.width+self.pad, self.pad))
            if y1pad > 0:
                bot_mirrored_dat = ImageOps.flip(self.inimg.crop((0, self.height-self.pad, self.width, self.height)))
                ret.paste(bot_mirrored_dat, (self.pad, self.height+self.pad, self.width+self.pad, self.height+self.pad*2))
            # useful info
            usefuldim = self.pad, self.pad, self.width+self.pad, self.height+self.pad
            usefulstart = self.pad, self.pad
        else:
            # x-y indices (0 to iperhl, 0 to ipervl)
            yi = int(math.ceil((i+1)/(self.iperhl+1) - 1))   # line number
            xi = i-yi*(self.iperhl+1)
            # x-y start-end position on fs image
            x0 = self.ucs * xi - self.ol * xi - self.pad
            x1 = x0+self.cs
            y0 = self.ucs * yi - self.ol * yi - self.pad
            y1 = y0+self.cs
            ret = Image.new('RGB', (self.cs, self.cs))
            # amount padded to have a cs x cs crop
            x0pad = -min(0, x0)
            x1pad = max(0, x1 - self.width)
            y0pad = -min(0, y0)
            y1pad = max(0, y1 - self.height)
            crop = self.inimg.crop((x0+x0pad, y0+y0pad, x1-x1pad, y1-y1pad))
            ret.paste(crop, (x0pad, y0pad, self.cs-x1pad, self.cs-y1pad))
            # mirror stuff:
            if x0pad > 0:
                # FIXME invert inverts the colors, not the coordinates
                left_mirrored_dat = ImageOps.mirror(self.inimg.crop((x0+x0pad, y0+y0pad, x0+x0pad*2, y1-y1pad)))
                ret.paste(left_mirrored_dat, (0, y0pad, x0pad, self.cs-y1pad))
            if x1pad > 0:
                right_mirrored_dat = ImageOps.mirror(self.inimg.crop((x1-x1pad*2, y0+y0pad, x1-x1pad, y1-y1pad)))
                ret.paste(right_mirrored_dat, (self.cs-x1pad, y0pad, self.cs, self.cs-y1pad))
            if y0pad > 0:
                top_mirrored_dat = ImageOps.flip(self.inimg.crop((x0+x0pad, y0+y0pad, x1-x1pad, y0+y0pad*2)))
                ret.paste(top_mirrored_dat, (x0pad, 0, self.cs-x1pad, y0pad))
            if y1pad > 0:
                bot_mirrored_dat = ImageOps.flip(self.inimg.crop((x0+x0pad, y1-y1pad*2, x1-x1pad, y1-y1pad)))
                ret.paste(bot_mirrored_dat, (x0pad, self.cs-y1pad, self.cs-x1pad, self.cs))
            # useful info
            usefuldim = (self.pad, self.pad, self.cs-max(self.pad,x1pad), self.cs-max(self.pad,y1pad))
            usefulstart = (x0+self.pad, y0+self.pad)
        return self.totensor(ret), torch.IntTensor(usefuldim), torch.IntTensor(usefulstart)
    def __len__(self):
        return self.size

#
# Standard import
#import importlib
# Load "module.submodule.MyClass"
#MyClass = getattr(importlib.import_module("module.submodule"), "MyClass")
# Instantiate the class (pass arguments to the constructor, if needed)
#instance = MyClass()

model = Model.instantiate_model(network=args.g_network, model_path=args.model_path,
                                strparameters=args.model_parameters, keyword='generator',
                                device=device, models_dpath=args.models_dpath)
model.eval()  # evaluation mode
model = model.to(device)
ds = OneImageDS(args.input, cs, ucs, args.overlap, whole_image=args.whole_image, pad=args.pad)
# multiple workers cannot access the same PIL object without crash
DLoader = DataLoader(dataset=ds, num_workers=0, drop_last=False, batch_size=args.batch_size, shuffle=False)
topil = torchvision.transforms.ToPILImage()
fswidth, fsheight = Image.open(args.input).size
newimg = torch.zeros(3, fsheight, fswidth, dtype=torch.float32)

def make_seamless_edges(tcrop, x0, y0):
    if x0 != 0:#left
        tcrop[:,:,0:args.overlap] = tcrop[:,:,0:args.overlap].div(2)
    if y0 != 0:#top
        tcrop[:,0:args.overlap,:] = tcrop[:,0:args.overlap,:].div(2)
    if x0 + ucs < fswidth and args.overlap:#right
        tcrop[:,:,-args.overlap:] = tcrop[:,:,-args.overlap:].div(2)
    if y0 + ucs < fsheight and args.overlap:#bottom
        tcrop[:,-args.overlap:,:] = tcrop[:,-args.overlap:,:].div(2)
    return tcrop

start_time = time.time()
for n_count, ydat in enumerate(DLoader):
        print(str(n_count)+'/'+str(int(len(ds)/args.batch_size)))
        ybatch, usefuldims, usefulstarts = ydat
        if args.max_subpixels is not None and math.prod(ybatch.shape) > args.max_subpixels:
            sys.exit(f'denoise_image.py: {ybatch.shape=}, {math.prod(ybatch.shape)=} > {args.max_subpixels=} for {args.input=}; aborting')
        ybatch = ybatch.to(device)
        xbatch = model(ybatch)
        torch.cuda.synchronize()
        for i in range(args.batch_size):
            ud = usefuldims[i]
            # pytorch represents images as [channels, height, width]
            # TODO test leaving on GPU longer
            tensimg = xbatch[i][:,ud[1]:ud[3], ud[0]:ud[2]].cpu().detach()
            if args.whole_image:
                newimg = tensimg
            else:
                absx0, absy0 = tuple(usefulstarts[i].tolist())
                tensimg = make_seamless_edges(tensimg, absx0, absy0)
                if args.debug:
                    os.makedirs('dbg', exist_ok=True)
                    torchvision.utils.save_image(xbatch[i], 'dbg/crop'+str(n_count)+'_'+str(i)+'_1.jpg')
                    torchvision.utils.save_image(tensimg, 'dbg/crop'+str(n_count)+'_'+str(i)+'_2.jpg')
                    print(tensimg.shape)
                    print((absx0,absy0,ud))
                newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]] = newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]].add(tensimg)
if args.debug:
    torchvision.utils.save_image(xbatch[i].cpu().detach(), args.output+'dbg_inclborders.tif')  # dbg: get img with borders
pt_helpers.tensor_to_imgfile(newimg, args.output)
if args.output[:-4] == '.jpg' and args.exif_method == 'piexif':
    piexif.transplant(args.input, args.output)
elif args.exif_method != 'noexif':
    cmd = ['exiftool', '-TagsFromFile', args.input, args.output, '-overwrite_original']
    subprocess.run(cmd)
print(f'Wrote denoised image to {args.output}')
print('Elapsed time: '+str(time.time()-start_time)+' seconds')
