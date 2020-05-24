# Denoise an image: the ultimately useful test. cs and ucs should be set depending on the network (ie UNet can handle bigger, *128Net should have cs=128 ucs=112). TODO make exif and such optional
import os
import argparse
import torchvision
import torch
from math import ceil
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
from nn_common import Model, default_values
import torch.backends.cudnn as cudnn
try:
	import piexif   # TODO make it optional
except ImportError:
	pass
import subprocess

default_cs = 128
default_ucs = 112
default_cs_unet = 256
default_ucs_unet = 192

# TODO handle CPU

parser = argparse.ArgumentParser(description='Image cropper with overlap')
parser.add_argument('--cs', type=int, help='Tile size (model was probably trained with 128, different values will work with unknown results)')
parser.add_argument('--ucs', type=int, help='Useful tile size (should be <=.75*cs for U-Net, a smaller value may result in less grid artifacts but costs computation time')
parser.add_argument('-ol', '--overlap', default=6, type=int, help='Merge crops with this much overlap (Reduces grid artifacts, may reduce sharpness between crops, costs computation time)')
parser.add_argument('-i', '--input', default='in.jpg', type=str, help='Input image file')
parser.add_argument('-o', '--output', default='out.tif', type=str, help='Output file with extension')
parser.add_argument('-b', '--batch_size', type=int, default=1)  # TODO >1 is broken
parser.add_argument('--debug', action='store_true', help='Debug (store all intermediate crops in ./dbg, display useful messages)')
parser.add_argument('--cuda_device', default=0, type=int, help='Device number (default: 0, typically 0-3]])')
parser.add_argument('--exif_method', default='piexif', type=str, help='How is exif data copied over? (piexif, exiftool, noexif)')
parser.add_argument('--network', type=str, default=default_values['g_network'], help='Generator network (default: %s)'%default_values['g_network'])
parser.add_argument('--model_path', help='Generator pretrained model path (.pth for model, .pt for dictionary)')
parser.add_argument('--model_parameters', type=str, help='Model parameters with format "parameter1=value1,parameter2=value2"')
args = parser.parse_args()

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
	
torch.cuda.set_device(args.cuda_device)
cudnn.benchmark = True

torch.manual_seed(123)
torch.cuda.manual_seed(123)

class OneImageDS(Dataset):
	def __init__(self, inimg, cs, ucs, ol):
		self.inimg = Image.open(inimg)
		self.width, self.height = self.inimg.size
		self.cs, self.ucs, self.ol = cs, ucs, ol	# crop size, useful crop size, overlap
		self.totensor = torchvision.transforms.ToTensor()
		self.iperhl = ceil((self.width - self.ucs) / (self.ucs - self.ol)) # i_per_hline, or crops per line
		self.pad = int((self.cs - self.ucs) / 2)
		ipervl = ceil((self.height - self.ucs) / (self.ucs - self.ol))
		self.size = (self.iperhl+1) * (ipervl+1)
	def __getitem__(self, i):
		# x-y indices (0 to iperhl, 0 to ipervl)
		yi = int(ceil((i+1)/(self.iperhl+1) - 1))   # line number
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
			left_mirrored_dat = ImageOps.invert(self.inimg.crop((x0+x0pad, y0+y0pad, x0+x0pad*2, y1-y1pad)))
			ret.paste(left_mirrored_dat, (0, y0pad, x0pad, self.cs-y1pad))
		if x1pad > 0:
			right_mirrored_dat = ImageOps.invert(self.inimg.crop((x1-x1pad*2, y0+y0pad, x1-x1pad, y1-y1pad)))
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

model = Model.instantiate_model(network=args.network, model_path=args.model_path, strparameters=args.model_parameters, keyword='generator')
model.eval()  # evaluation mode
if torch.cuda.is_available():
	model = model.cuda()
ds = OneImageDS(args.input, cs, ucs, args.overlap)
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
		ybatch = ybatch.cuda()
		xbatch = model(ybatch)
		torch.cuda.synchronize()
		for i in range(args.batch_size):
			ud = usefuldims[i]
			# pytorch represents images as [channels, height, width]
			# TODO test leaving on GPU longer
			tensimg = xbatch[i][:,ud[1]:ud[3], ud[0]:ud[2]].cpu().detach()
			absx0, absy0 = tuple(usefulstarts[i].tolist())
			tensimg = make_seamless_edges(tensimg, absx0, absy0)
			if args.debug:
				os.makedirs('dbg', exist_ok=True)
				torchvision.utils.save_image(xbatch[i], 'dbg/crop'+str(n_count)+'_'+str(i)+'_1.jpg')
				torchvision.utils.save_image(tensimg, 'dbg/crop'+str(n_count)+'_'+str(i)+'_2.jpg')
				print(tensimg.shape)
				print((absx0,absy0,ud))
			newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]] = newimg[:,absy0:absy0+tensimg.shape[1],absx0:absx0+tensimg.shape[2]].add(tensimg)
torchvision.utils.save_image(newimg, args.output)
if args.output[:-4] == '.jpg' and args.exif_method == 'piexif':
	piexif.transplant(args.input, args.output)
elif args.exif_method != 'noexif':
	cmd = ['exiftool', '-TagsFromFile', args.input, args.output, '-overwrite_original']
	subprocess.run(cmd)
print('Elapsed time: '+str(time.time()-start_time)+' seconds')
