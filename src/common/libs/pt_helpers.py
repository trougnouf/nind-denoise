import torch
import cv2
from PIL import Image
import torchvision
import numpy as np
import sys
sys.path.append('..')
from common.libs import np_imgops
from common.libs import pt_losses

def fpath_to_tensor(img_fpath, device=torch.device(type='cpu'), batch=False):
    #totensor = torchvision.transforms.ToTensor()
    #pilimg = Image.open(imgpath).convert('RGB')
    #return totensor(pilimg)  # replaced w/ opencv to handle >8bits
    tensor = torch.tensor(np_imgops.img_path_to_np_flt(img_fpath), device=device)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor

def tensor_to_imgfile(tensor, path):
    if tensor.dtype == torch.float32:
        if path[-4:].lower() in ['.jpg', 'jpeg']:  # 8-bit
            return torchvision.utils.save_image(tensor.clip(0,1), path)
        elif path[-4:].lower() in ['.png', '.tif', 'tiff']:  # 16-bit
            nptensor = (tensor.clip(0,1)*65535).round().cpu().numpy().astype(np.uint16).transpose(1,2,0)
            nptensor = cv2.cvtColor(nptensor, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, nptensor)
        else:
            raise NotImplementedError(f'Extension in {path}')
    elif tensor.dtype == torch.uint8:
        tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
        pilimg = Image.fromarray(tensor)
        pilimg.save(path)
    else:
        raise NotImplementedError(tensor.dtype)
    
def get_losses(img1_fpath, img2_fpath):
    img1 = fpath_to_tensor(img1_fpath).unsqueeze(0)
    img2 = fpath_to_tensor(img2_fpath).unsqueeze(0)
    assert img1.shape == img2.shape, f'{img1.shape=}, {img2.shape=}'
    res = dict()
    res['mse'] = torch.nn.functional.mse_loss(img1, img2).item()
    res['ssim'] = pt_losses.SSIM_loss()(img1, img2).item()
    res['msssim'] = pt_losses.MS_SSIM_loss()(img1, img2).item()
    return res

def get_device(device_n=None):
    """get device given index (-1 = CPU)"""
    if isinstance(device_n, torch.device):
        return device_n
    elif isinstance(device_n, str):
        return torch.device(device_n)
    if device_n is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print('get_device: cuda not available; defaulting to cpu')
            return torch.device("cpu")
    elif torch.cuda.is_available() and device_n >= 0:
        return torch.device("cuda:%i" % device_n)
    elif device_n >= 0:
        print('get_device: cuda not available')
    return torch.device('cpu')
    
