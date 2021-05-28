# -*- coding: utf-8 -*-

import torch
import piqa

class MS_SSIM_loss(piqa.MS_SSIM):
    def __init__(self, **kwargs):
        r""""""
        super().__init__(reduction=None, **kwargs)
    def forward(self, input, target):
        return 1-super().forward(input, target)
    
class SSIM_loss(piqa.SSIM):
    def __init__(self, **kwargs):
        r""""""
        super().__init__(reduction=None, **kwargs)
    def forward(self, input, target):
        return 1-super().forward(input, target)
        
if __name__ == '__main__':
    def findvaliddim(start):
        try:
            piqa.MS_SSIM()(torch.rand(1,3,start,start),torch.rand(1,3,start,start))
            print(start)
            return(start)
        except RuntimeError:
            print(start)
            findvalid(start+1)
    findvaliddim(1)  # result is 162