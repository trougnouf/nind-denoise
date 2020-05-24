import torch.nn as nn
import torch

# 128 PReLU generator
#size is determined by ((min+8)×3+8)×3+14
#therefore input resolution can be 119+x*9
class Hulb128Net(nn.Module):
    def __init__(self, funit=32, activation='PReLU'):
        super(Hulb128Net, self).__init__()
        funit=int(funit)
        self.enc128to126std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc126to122std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec122to126std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec122to126dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec126to128std = nn.Sequential(
            # in: 2+6
            nn.ConvTranspose2d(8*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(2*funit, 3, 1),
        )
        if activation is None or activation == 'None':
            self.activation = None
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU(init=0.01)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            print('Error: unknown activation (%s)' % activation)

    def forward(self, x):
        # down
        # 160 to 150
        l126 = self.enc128to126std(x)
        l122 = torch.cat([self.enc126to122std(l126), self.enc126to122dil(l126)], 1)
        l118 = torch.cat([self.enc122to118std(l122), self.enc122to118dil(l122), self.enc128to118dil(x)], 1)
        del(x)
        l114 = torch.cat([self.enc118to114std(l118), self.enc118to114dil(l118)], 1)
        l38 = self.enc114to38str(l114)
        l34 = torch.cat([self.enc38to34std(l38), self.enc38to34dil(l38)], 1)
        l30 = torch.cat([self.enc34to30std(l34), self.enc34to30dil(l34)], 1)
        l10 = self.enc30to10str(l30)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l2 = torch.cat([self.enc6to2std(l6), self.enc6to2dil(l6)], 1)
        # up
        l6 = torch.cat([l6, self.dec2to6std(l2), self.dec2to6dil(l2)], 1)
        del(l2)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l30 = torch.cat([l30, self.dec10to30str(l10)], 1)
        del(l10)
        l34 = torch.cat([l34, self.dec30to34std(l30), self.dec30to34dil(l30)], 1)
        del(l30)
        l38 = torch.cat([l38, self.dec34to38std(l34), self.dec34to38dil(l34)], 1)
        del(l34)
        l114 = torch.cat([l114, self.dec38to114str(l38)], 1)
        del(l38)
        l118 = torch.cat([l118, self.dec114to118std(l114), self.dec114to118dil(l114)], 1)
        del(l114)
        l122 = torch.cat([l122, self.dec118to122std(l118), self.dec118to122dil(l118)], 1)
        del(l118)
        l126 = torch.cat([l126, self.dec122to126std(l122), self.dec122to126dil(l122)], 1)
        res = self.dec126to128std(l126)
        if self.activation is None:
            return res
        else:
            return self.activation(res)


#112 PReLU w/ BN discriminator
# w/ Hul128Net BS 12 on 7GB GPU, 20 on 11GB GPU or 19 if conditional
class Hul112Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False, out_activation = 'PReLU'):
        super(Hul112Disc, self).__init__()
        self.funit = funit
        self.enc112to108std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc108to104std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc112to108dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc108to104dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc104to102std = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc112to102dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc102to34str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc18to6str = nn.Sequential(
            nn.Conv2d(8*funit, 8*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(8*funit),
        )


        if not finalpool:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 1*funit, 2),
                nn.PReLU(init=0.01),
                nn.Conv2d(1*funit, 1, 1),
                #nn.Sigmoid()
            )
        else:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 2*funit, 1),
                nn.PReLU(init=0.01),
                nn.Conv2d(2*funit, 1, 1),
                #nn.Sigmoid(),
                nn.AdaptiveMaxPool2d(1),
            )
        if out_activation is None or out_activation == 'None':
            self.out_activation = None
        elif out_activation == 'PReLU':
            self.out_activation = nn.PReLU(init=0.01)
        elif out_activation == 'Sigmoid':
            self.out_activation = nn.Sigmoid()
        elif out_activation == 'LeakyReLU':
            self.out_activation = nn.LeakyReLU()
    def forward(self, x):
        layer = torch.cat([self.enc112to108std(x), self.enc112to108dil(x)], 1)
        layer = torch.cat([self.enc108to104std(layer), self.enc108to104dil(layer)], 1)
        layer = torch.cat([self.enc104to102std(layer), self.enc112to102dil(x)], 1)
        layer = self.enc102to34str(layer)
        layer = torch.cat([self.enc34to30std(layer), self.enc34to30dil(layer)], 1)
        layer = torch.cat([self.enc30to26std(layer), self.enc30to26dil(layer)], 1)
        layer = torch.cat([self.enc26to22std(layer), self.enc26to22dil(layer)], 1)
        layer = torch.cat([self.enc22to18std(layer), self.enc22to18dil(layer)], 1)
        layer = self.enc18to6str(layer)
        layer = torch.cat([self.enc6to2std(layer), self.enc6to2dil(layer)], 1)
        layer = self.decide(layer)
        if self.out_activation is not None:
            return self.out_activation(layer)
        return layer

# SELU generator
# not tested
class Hulbs128Net(nn.Module):
    def __init__(self, funit=32, activation='PReLU'):
        super(Hulbs128Net, self).__init__()
        self.enc128to126std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.SELU(),
        )
        self.enc126to122std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc122to118std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.SELU(),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.SELU(),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.SELU(),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.SELU(),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.SELU(),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.SELU(),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.SELU(),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec122to126std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.SELU(),
        )
        self.dec122to126dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.SELU(),
        )
        self.dec126to128std = nn.Sequential(
            # in: 2+6
            nn.ConvTranspose2d(8*funit, 2*funit, 3),
            nn.SELU(),
            nn.ConvTranspose2d(2*funit, 3, 1),
        )
        if activation is None or activation == 'None':
            self.activation = None
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'PReLU':
            self.activation = PReLU(init=0.01)
        elif activation == 'SELU':
            self.activation = SELU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # down
        # 160 to 150
        l126 = self.enc128to126std(x)
        l122 = torch.cat([self.enc126to122std(l126), self.enc126to122dil(l126)], 1)
        l118 = torch.cat([self.enc122to118std(l122), self.enc122to118dil(l122), self.enc128to118dil(x)], 1)
        del(x)
        l114 = torch.cat([self.enc118to114std(l118), self.enc118to114dil(l118)], 1)
        l38 = self.enc114to38str(l114)
        l34 = torch.cat([self.enc38to34std(l38), self.enc38to34dil(l38)], 1)
        l30 = torch.cat([self.enc34to30std(l34), self.enc34to30dil(l34)], 1)
        l10 = self.enc30to10str(l30)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l2 = torch.cat([self.enc6to2std(l6), self.enc6to2dil(l6)], 1)
        # up
        l6 = torch.cat([l6, self.dec2to6std(l2), self.dec2to6dil(l2)], 1)
        del(l2)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l30 = torch.cat([l30, self.dec10to30str(l10)], 1)
        del(l10)
        l34 = torch.cat([l34, self.dec30to34std(l30), self.dec30to34dil(l30)], 1)
        del(l30)
        l38 = torch.cat([l38, self.dec34to38std(l34), self.dec34to38dil(l34)], 1)
        del(l34)
        l114 = torch.cat([l114, self.dec38to114str(l38)], 1)
        del(l38)
        l118 = torch.cat([l118, self.dec114to118std(l114), self.dec114to118dil(l114)], 1)
        del(l114)
        l122 = torch.cat([l122, self.dec118to122std(l118), self.dec118to122dil(l118)], 1)
        del(l118)
        l126 = torch.cat([l126, self.dec122to126std(l122), self.dec122to126dil(l122)], 1)
        res = self.dec126to128std(l126)
        if self.activation is None:
            return res
        else:
            return self.activation(res)


#112 PReLU w/ BN discriminator
# w/ Hul128Net BS 12 on 7GB GPU, 20 on 11GB GPU or 19 if conditional
class Hulb112Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False, out_activation = 'PReLU'):
        super(Hulb112Disc, self).__init__()
        self.funit = funit
        self.enc112to108std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc108to104std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc112to108dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc108to104dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc104to102std = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc112to102dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc102to34str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc30to26std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc26to22std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc22to18std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )

        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc30to26dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc26to22dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc22to18dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
        )

        self.enc18to6str = nn.Sequential(
            nn.Conv2d(8*funit, 8*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
        )


        if not finalpool:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 1*funit, 2),
                nn.PReLU(init=0.01),
                nn.Conv2d(1*funit, 1, 1),
                #nn.Sigmoid()
            )
        else:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 2*funit, 1),
                nn.PReLU(init=0.01),
                nn.Conv2d(2*funit, 1, 1),
                #nn.Sigmoid(),
                nn.AdaptiveMaxPool2d(1),
            )
        if out_activation is None or out_activation == 'None':
            self.out_activation = None
        elif out_activation == 'PReLU':
            self.out_activation = nn.PReLU(init=0.01)
        elif out_activation == 'Sigmoid':
            self.out_activation = nn.Sigmoid()
    def forward(self, x):
        layer = torch.cat([self.enc112to108std(x), self.enc112to108dil(x)], 1)
        layer = torch.cat([self.enc108to104std(layer), self.enc108to104dil(layer)], 1)
        layer = torch.cat([self.enc104to102std(layer), self.enc112to102dil(x)], 1)
        layer = self.enc102to34str(layer)
        layer = torch.cat([self.enc34to30std(layer), self.enc34to30dil(layer)], 1)
        layer = torch.cat([self.enc30to26std(layer), self.enc30to26dil(layer)], 1)
        layer = torch.cat([self.enc26to22std(layer), self.enc26to22dil(layer)], 1)
        layer = torch.cat([self.enc22to18std(layer), self.enc22to18dil(layer)], 1)
        layer = self.enc18to6str(layer)
        layer = torch.cat([self.enc6to2std(layer), self.enc6to2dil(layer)], 1)
        layer = self.decide(layer)
        if self.out_activation is not None:
            return self.out_activation(layer)
        return layer

#112 PReLU w/ BN discriminator
# w/ Hul128Net BS 12 on 7GB GPU, 20 on 11GB GPU or 19 if conditional
class Hull112Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False, out_activation = 'PReLU'):
        super(Hull112Disc, self).__init__()
        self.funit = funit
        self.enc112to108std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.LeakyReLU(),
            nn.Conv2d(2*funit, 2*funit, 3),
            nn.LeakyReLU(),
        )
        self.enc108to104std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*funit),
        )
        self.enc112to108dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=2),
            nn.LeakyReLU(),
        )
        self.enc108to104dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*funit),
        )
        self.enc104to102std = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc112to102dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*funit),
        )
        self.enc102to34str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(6*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18std = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )

        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False, dilation=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc30to26dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc26to22dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )
        self.enc22to18dil = nn.Sequential(
            nn.Conv2d(8*funit, 4*funit, 3, bias=False, dilation=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*funit),
        )

        self.enc18to6str = nn.Sequential(
            nn.Conv2d(8*funit, 8*funit, 3, stride=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8*funit),
        )


        if not finalpool:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.LeakyReLU(),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.LeakyReLU(),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.LeakyReLU(),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 1*funit, 2),
                nn.LeakyReLU(),
                nn.Conv2d(1*funit, 1, 1),
                #nn.Sigmoid()
            )
        else:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(8*funit, 6*funit, 3, bias=False),
                nn.LeakyReLU(),
                nn.BatchNorm2d(6*funit),
                nn.Conv2d(6*funit, 3*funit, 3, bias=False),
                nn.LeakyReLU(),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(8*funit, 3*funit, 3, bias=False, dilation=2),
                nn.LeakyReLU(),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(6*funit, 2*funit, 1),
                nn.LeakyReLU(),
                nn.Conv2d(2*funit, 1, 1),
                #nn.Sigmoid(),
                nn.AdaptiveMaxPool2d(1),
            )
        if out_activation is None or out_activation == 'None':
            self.out_activation = None
        elif out_activation == 'PReLU':
            self.out_activation = PReLU(init=0.01)
        elif out_activation == 'Sigmoid':
            self.out_activation = nn.Sigmoid()
        elif out_activation == 'LeakyReLU':
            self.out_activation = nn.LeakyReLU()
    def forward(self, x):
        layer = torch.cat([self.enc112to108std(x), self.enc112to108dil(x)], 1)
        layer = torch.cat([self.enc108to104std(layer), self.enc108to104dil(layer)], 1)
        layer = torch.cat([self.enc104to102std(layer), self.enc112to102dil(x)], 1)
        layer = self.enc102to34str(layer)
        layer = torch.cat([self.enc34to30std(layer), self.enc34to30dil(layer)], 1)
        layer = torch.cat([self.enc30to26std(layer), self.enc30to26dil(layer)], 1)
        layer = torch.cat([self.enc26to22std(layer), self.enc26to22dil(layer)], 1)
        layer = torch.cat([self.enc22to18std(layer), self.enc22to18dil(layer)], 1)
        layer = self.enc18to6str(layer)
        layer = torch.cat([self.enc6to2std(layer), self.enc6to2dil(layer)], 1)
        layer = self.decide(layer)
        if self.out_activation is not None:
            return self.out_activation(layer)
        return layer

#112 PReLU w/ BN discriminator
# w/ Hul128Net BS 12 on 7GB GPU, 20 on 11GB GPU or 19 if conditional
class Hulf112Disc(nn.Module):
    def __init__(self, input_channels = 6, funit = 32, finalpool = False, out_activation = 'PReLU'):
        super(Hulf112Disc, self).__init__()
        self.funit = funit
        self.enc112to108std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc108to104std = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc112to108dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=2),
            nn.PReLU(init=0.01),
        )
        self.enc108to104dil = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc104to102std = nn.Sequential(
            nn.Conv2d(8*funit, 8*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(8*funit),
        )
        self.enc112to102dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc102to34str = nn.Sequential(
            nn.Conv2d(10*funit, 10*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(10*funit, 10*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
            nn.Conv2d(10*funit, 10*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc30to26std = nn.Sequential(
            nn.Conv2d(20*funit, 20*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(20*funit),
            nn.Conv2d(20*funit, 10*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc26to22std = nn.Sequential(
            nn.Conv2d(20*funit, 20*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(20*funit),
            nn.Conv2d(20*funit, 10*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc22to18std = nn.Sequential(
            nn.Conv2d(20*funit, 20*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(20*funit),
            nn.Conv2d(20*funit, 12*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(12*funit),
        )

        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(10*funit, 10*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc30to26dil = nn.Sequential(
            nn.Conv2d(20*funit, 10*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc26to22dil = nn.Sequential(
            nn.Conv2d(20*funit, 10*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(10*funit),
        )
        self.enc22to18dil = nn.Sequential(
            nn.Conv2d(20*funit, 12*funit, 3, bias=False, dilation=2),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(12*funit),
        )

        self.enc18to6str = nn.Sequential(
            nn.Conv2d(24*funit, 24*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(24*funit),
        )


        if not finalpool:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(24*funit, 24*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(24*funit),
                nn.Conv2d(24*funit, 12*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(24*funit, 12*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(24*funit, 6*funit, 2),
                nn.PReLU(init=0.01),
                nn.Conv2d(6*funit, 1, 1),
                #nn.Sigmoid()
            )
        else:
            self.enc6to2std = nn.Sequential(
                nn.Conv2d(24*funit, 24*funit, 3, bias=False),
                nn.PReLU(init=0.01),
                nn.BatchNorm2d(24*funit),
                nn.Conv2d(24*funit, 6*funit, 3, bias=False),
                nn.PReLU(init=0.01),
            )
            self.enc6to2dil = nn.Sequential(
                nn.Conv2d(24*funit, 6*funit, 3, bias=False, dilation=2),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(12*funit, 6*funit, 1),
                nn.PReLU(init=0.01),
                nn.Conv2d(6*funit, 1, 1),
                #nn.Sigmoid(),
                nn.AdaptiveMaxPool2d(1),
            )
        if out_activation is None or out_activation == 'None':
            self.out_activation = None
        elif out_activation == 'PReLU':
            self.out_activation = nn.PReLU(init=0.01)
        elif out_activation == 'Sigmoid':
            self.out_activation = nn.Sigmoid()
        elif out_activation == 'LeakyReLU':
            self.out_activation = nn.LeakyReLU()
    def forward(self, x):
        layer = torch.cat([self.enc112to108std(x), self.enc112to108dil(x)], 1)
        layer = torch.cat([self.enc108to104std(layer), self.enc108to104dil(layer)], 1)
        layer = torch.cat([self.enc104to102std(layer), self.enc112to102dil(x)], 1)
        layer = self.enc102to34str(layer)
        layer = torch.cat([self.enc34to30std(layer), self.enc34to30dil(layer)], 1)
        layer = torch.cat([self.enc30to26std(layer), self.enc30to26dil(layer)], 1)
        layer = torch.cat([self.enc26to22std(layer), self.enc26to22dil(layer)], 1)
        layer = torch.cat([self.enc22to18std(layer), self.enc22to18dil(layer)], 1)
        layer = self.enc18to6str(layer)
        layer = torch.cat([self.enc6to2std(layer), self.enc6to2dil(layer)], 1)
        layer = self.decide(layer)
        if self.out_activation is not None:
            return self.out_activation(layer)
        return layer
