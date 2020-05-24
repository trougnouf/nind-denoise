import torch.nn as nn
import torch

# WIP barely started

class DDnet(nn.Module):
    def __init__(self, funit=32, activation='PReLU', lv=16):
        super(DDnet, self).__init__()
        self.lv = lv
        if activation == 'PReLU':
 mm``
        self.padder = nn.ReplicationPad2d(2)
        def make_init_conv(lv):
            return nn.Sequential(
                nn.Conv2d(3, funit, 3, dilation=lv),
                self.activation
            )
        self.conv_init = [make_init_conv(i+1) for i in range(lv)]
    def forward(self, y):
        y = self.padder(y)
        c1 = [conv(y) for conv in self.conv_init]
        
