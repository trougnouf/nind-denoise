'''
Image cropper with overlap

Denoise an image: the ultimately useful test. cs and ucs should be set
depending on the network

TODO make exif and such optional
TODO 16-bit input! (replace PIL with numpy)

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
try:
    import piexif   # TODO make it optional
except ImportError:
    pass
import subprocess
import numpy as np
import sys
sys.path.append('..')
from common.libs import pt_helpers
from common.libs import utilities
from nind_denoise import nn_common
from common.libs import np_imgops
CS_UNET, UCS_UNET = 440, 320
CS_UTNET, UCS_UTNET = 504, 480
CS_UNK, UCS_UNK = 512, 448


def make_output_fpath(input_fpath, model_fpath):
    model_dpath = utilities.get_root(model_fpath)
    model_fn = utilities.get_leaf(model_fpath)
    img_fn = utilities.get_leaf(input_fpath)
    os.makedirs(os.path.join(model_dpath, 'test', 'denoised_images'), exist_ok=True)
    return os.path.join(model_dpath, 'test', 'denoised_images', f'{img_fn}_{model_fn}.tif')


def autodetect_network_cs_ucs(args) -> None:
    if args.g_network is None:
        print('network parameter not specified')
        if 'unet' in args.model_path.lower():
            args.g_network = 'UNet'
        elif 'utnet' in args.model_path.lower():
            args.g_network = 'UtNet'
        else:
            exit('Could not determine network architecture from path. Please specify a "--network" type (typically UNet or UtNet)')
        print(f'Assuming {args.g_network} from path')
    if args.cs is None or args.ucs is None:
        print('cs and/or ucs not set, using defaults ...')
        if args.g_network == 'UNet':
            args.cs = CS_UNET
            args.ucs = UCS_UNET
        elif args.g_network == 'UtNet':
            args.cs, args.ucs = CS_UTNET, UCS_UTNET
        else:
            print('Warning: cs and ucs not known for this architecture; values may be sub-optimal')
            args.cs, args.ucs = CS_UNK, UCS_UNK
        print(f'cs={args.cs}, ucs={args.ucs}')

class OneImageDS(Dataset):
    '''
    PyTorch compatible single-image dataset which crops an image into equal pieces with overlap and
    adds mirror padding to the edges
    Uses opencv to load images in 8/16 bits depth, and numpy operations
    Image format is always CxHxW
    '''
    def __init__(self, inimg_fpath, cs, ucs, ol, whole_image=False, pad=None):
        self.inimg = np_imgops.img_path_to_np_flt(inimg_fpath)
        self.width, self.height = self.inimg.shape[2], self.inimg.shape[1]
        if whole_image:
            self.pad = pad
            if self.pad is None or self.pad == 0:
                self.pad = 0
                print('OneImageDS: Warning: you should really consider (pad>0)')
            self.whole_image = True
            self.size = 1
        else:
            self.whole_image = False
            self.cs, self.ucs, self.ol = cs, ucs, ol    # crop size, useful crop size, overlap
            self.iperhl = math.ceil((self.width - self.ucs) / (self.ucs - self.ol)) # i_per_hline, or crops per line
            self.pad = int((self.cs - self.ucs) / 2)
            ipervl = math.ceil((self.height - self.ucs) / (self.ucs - self.ol))
            self.size = (self.iperhl+1) * (ipervl+1)
            if self.pad is None or self.pad == 0:
                self.pad = 0
                print('OneImageDS: Warning: you should really consider padding (cs>ucs)')
    def __getitem__(self, i):
        # mirrorring can be improved by including the corners
        if self.whole_image:
            xi = yi = x0 = y0 = 0
            x1, y1 = self.width, self.height
            ret = np.zeros((3, x1+self.pad*2, y1+self.pad*2), dtype=np.float32)
            crop = self.inimg
            # copy image to the center
            ret[:, self.pad:self.height+self.pad, self.pad:self.width+self.pad] = crop
            # mirror sides
            if self.pad:
                # left
                ret[:, self.pad:-self.pad, :self.pad] = np.flip(crop[:, :, :self.pad], axis=2)
                # right
                ret[:, self.pad:-self.pad, self.width+self.pad:] = np.flip(crop[:, :, self.width-self.pad:], axis=2)
                # top
                ret[:, :self.pad, self.pad:-self.pad] = np.flip(crop[:, :self.pad, :], axis=1)
                # bottom
                ret[:, self.height+self.pad:, self.pad:-self.pad] = np.flip(crop[:, self.height-self.pad:, :], axis=1)
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
            ret = np.ndarray((3, self.cs, self.cs), dtype=np.float32)
            # amount padded to have a cs x cs crop
            x0pad = -min(0, x0)
            x1pad = max(0, x1 - self.width)
            y0pad = -min(0, y0)
            y1pad = max(0, y1 - self.height)
            # determine crop of interest
            crop = self.inimg[:, y0+y0pad:y1-y1pad, x0+x0pad:x1-x1pad]
            # copy crop to center
            ret[:, y0pad:self.cs-y1pad, x0pad:self.cs-x1pad] = crop
            # mirror stuff:
            # this somewhat suspicious stuff can be tested (visualized crops) with the
            # --debu argument
            if x0pad > 0:  # left
                ret[:, y0pad:self.cs-y1pad, :x0pad] = np.flip(
                    self.inimg[:, y0+y0pad:y1-y1pad, x0+x0pad:x0+x0pad*2], axis=2)
                if y0pad > 0:  # top-left
                    ret[:, :y0pad, :x0pad] = np.flip(self.inimg[:, :y0pad, :x0pad], axis=(1,2))
                if y1pad > 0:  # bottom-left
                    ret[:, -y1pad:, :x0pad] = np.flip(self.inimg[:, -y1pad:, :x0pad], axis=(1,2))
            if x1pad > 0:  # right
                ret[:, y0pad:self.cs-y1pad, self.cs-x1pad:, ] = np.flip(
                    self.inimg[:, y0+y0pad:y1-y1pad, x1-x1pad*2:x1-x1pad], axis=2)
                if y0pad > 0:  # top-right
                    ret[:, :y0pad, -x1pad:] = np.flip(self.inimg[:, :y0pad, -x1pad:], axis=(1, 2))
                if y1pad > 0:  # bottom-right
                    ret[:, -y1pad:, -x1pad:] = np.flip(self.inimg[:, -y1pad:, -x1pad:], axis=(1, 2))
            if y0pad > 0:
                ret[:, :y0pad, x0pad:self.cs-x1pad] = np.flip(
                    self.inimg[:, y0+y0pad:y0+y0pad*2, x0+x0pad:x1-x1pad], axis=1)
            if y1pad > 0:
                ret[:, self.cs-y1pad:, x0pad:self.cs-x1pad] = np.flip(
                    self.inimg[:, y1-y1pad*2:y1-y1pad, x0+x0pad:x1-x1pad], axis=1)
            # useful info
            usefuldim = (self.pad, self.pad, self.cs-max(self.pad,x1pad), self.cs-max(self.pad,y1pad))
            usefulstart = (x0+self.pad, y0+self.pad)
        return torch.tensor(ret), torch.IntTensor(usefuldim), torch.IntTensor(usefulstart)
    
    def __len__(self):
        return self.size

if __name__ == '__main__':

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
    def argdevice(s):
        try:
            ret = int(s)
            if ret < 0:
                ret = 'cpu'
        except ValueError:
            ret = s
        return ret
    parser.add_argument('--device', dest='cuda_device', default=0, type=argdevice, help='Device number or id. If a number, id of the cuda GPU, otherwise name of the device (default: 0)')
    parser.add_argument('--exif_method', default='piexif', type=str, help='How is exif data copied over? (piexif, exiftool, noexif)')
    parser.add_argument('--g_network', '--network', '--arch', type=str, help='Generator network (typically UNet or UtNet)')
    parser.add_argument('--model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary), required')
    parser.add_argument('--model_parameters', type=str, help='Model parameters with format "parameter1=value1,parameter2=value2"')
    parser.add_argument('--max_subpixels', type=int, help='Max. number of sub-pixels, abort if exceeded.')
    parser.add_argument('--whole_image', action='store_true', help='Ignore cs and ucs, denoise whole image')
    parser.add_argument('--pad', type=int, help='Padding amt per side, only used for whole image (otherwise (cs-ucs)/2')
    parser.add_argument('--models_dpath', help='Directory where all models are saved (used when a model name is provided as model_path)')
    
    args, _ = parser.parse_known_args()
    assert args.model_path is not None
    autodetect_network_cs_ucs(args)
    
    def make_seamless_edges(tcrop, x0, y0):
        if x0 != 0:#left
            tcrop[:,:,0:args.overlap] = tcrop[:,:,0:args.overlap].div(2)
        if y0 != 0:#top
            tcrop[:,0:args.overlap,:] = tcrop[:,0:args.overlap,:].div(2)
        if x0 + args.ucs < fswidth and args.overlap:#right
            tcrop[:,:,-args.overlap:] = tcrop[:,:,-args.overlap:].div(2)
        if y0 + args.ucs < fsheight and args.overlap:#bottom
            tcrop[:,-args.overlap:,:] = tcrop[:,-args.overlap:,:].div(2)
        return tcrop

        if isinstance(args.cuda_device, int) and args.cuda_device >= 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(args.cuda_device)
                torch.backends.cudnn.benchmark = True
                torch.cuda.manual_seed(123)
            else:
                print("warning: PyTorch is not installed with CUDA. Defaulting to CPU.")
    torch.manual_seed(123)
    device = pt_helpers.get_device(args.cuda_device)
    if args.output is None:
        args.output = make_output_fpath(args.input, args.model_path)

    # ugly hardcoded hack for now
    if args.model_parameters is None and 'activation' in args.model_path:
        args.model_parameters = f"activation={args.model_path.split('activation')[-1].split('_')[1].split('_')[0]}"
        print(f'set model_parameters to {args.model_parameters} based on model_path')
    model = Model.instantiate_model(network=args.g_network, model_path=args.model_path,
                                    strparameters=args.model_parameters, keyword='generator',
                                    device=device, models_dpath=args.models_dpath)
    model.eval()  # evaluation mode
    model = model.to(device)
    ds = OneImageDS(args.input, args.cs, args.ucs, args.overlap, whole_image=args.whole_image, pad=args.pad)
    DLoader = DataLoader(dataset=ds,
                         num_workers=0 if args.batch_size == 1 else max(min(args.batch_size, os.cpu_count()//4), 1),
                         drop_last=False, batch_size=args.batch_size, shuffle=False)
    topil = torchvision.transforms.ToPILImage()
    fswidth, fsheight = Image.open(args.input).size
    newimg = torch.zeros(3, fsheight, fswidth, dtype=torch.float32)

    start_time = time.time()
    for n_count, ydat in enumerate(DLoader):
        print(str(n_count)+'/'+str(int(len(ds)/args.batch_size)))
        ybatch, usefuldims, usefulstarts = ydat
        if args.max_subpixels is not None and math.prod(ybatch.shape) > args.max_subpixels:
            sys.exit(f'denoise_image.py: {ybatch.shape=}, {math.prod(ybatch.shape)=} > {args.max_subpixels=} for {args.input=}; aborting')
        ybatch = ybatch.to(device)
        xbatch = model(ybatch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for i in range(ybatch.size(0)):
            ud = usefuldims[i]
            # pytorch represents images as [channels, height, width]
            # TODO test leaving on GPU longer
            # TODO reconstruct image with batch_size > 1
            tensimg = xbatch[i][:,ud[1]:ud[3], ud[0]:ud[2]].cpu().detach()
            if args.whole_image:
                newimg = tensimg
            else:
                absx0, absy0 = tuple(usefulstarts[i].tolist())
                tensimg = make_seamless_edges(tensimg, absx0, absy0)
                if args.debug:
                    os.makedirs('dbg', exist_ok=True)
                    torchvision.utils.save_image(xbatch[i], 'dbg/crop'+str(n_count)+'_'+str(i)+'_denoised.jpg')
                    torchvision.utils.save_image(tensimg, 'dbg/crop'+str(n_count)+'_'+str(i)+'_tensimg.jpg')
                    torchvision.utils.save_image(ybatch[i], 'dbg/crop'+str(n_count)+'_'+str(i)+'_noisy.jpg')
                    print(tensimg.shape)
                    print((absx0,absy0,ud))
                newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]] = newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]].add(tensimg)
    if args.debug:
        torchvision.utils.save_image(xbatch[i].cpu().detach(), args.output+'dbg_inclborders.tif')  # dbg: get img with borders
    pt_helpers.tensor_to_imgfile(newimg, args.output)
    print(f'Denoised image written to {args.output}')
    if args.output[:-4] == '.jpg' and args.exif_method == 'piexif':
        piexif.transplant(args.input, args.output)
    elif args.exif_method != 'noexif':
        cmd = ['exiftool', '-TagsFromFile', args.input, args.output, '-all', '-icc_profile', '-overwrite_original']
        subprocess.run(cmd)
    print(f'Wrote denoised image to {args.output}')
    print('Elapsed time: '+str(time.time()-start_time)+' seconds')
