import torch.nn as nn

# based on https://arxiv.org/pdf/1603.05027.pdf ish
class RedishCNN(nn.Module):
    def __init__(self, n_channels=128, image_channels=3, kernel_size=5, depth=30, find_noise = False):
        super(RedishCNN, self).__init__()
        self.depth = depth
        self.bn = nn.BatchNorm2d(n_channels)
        self.conv_first = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.relu = nn.RReLU()
        self.find_noise = find_noise

    def forward(self, x):
        residuals = []
        layer = self.relu(self.conv_first(x))   #c1
        residuals.append(layer.clone())
        for _ in range(int(floor(self.depth-6)/2)):
            for _ in range(2):
                layer = self.bn(layer)
                layer = self.relu(layer)
                layer = self.conv(layer)
            residuals.append(layer.clone())
        layer = self.relu(self.conv(layer))     #clast
        layer = self.relu(self.deconv(layer))   #d1
        layer = self.relu(layer+residuals.pop())
        for _ in range(int(floor(self.depth-6)/2)):
            for _ in range(2):
                layer = self.bn(layer)
                layer = self.relu(layer)
                layer = self.deconv(layer)
            layer = self.relu(layer+residuals.pop())
        layer = self.relu(self.deconv_last(layer))
        if self.find_noise:
            return x - layer
        else:
            return layer


# 256-px
class HunkyDisc(nn.Module):
    def __init__(self, input_channels):
        super(HunkyDisc, self).__init__()
        #256
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 248
            nn.MaxPool2d(2),
            # 124
            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # 120
            nn.MaxPool2d(2),
            nn.Conv2d(96,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 56
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 24
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 8
            nn.MaxPool2d(2),
            nn.Conv2d(512,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, 2),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.enc(x)

class HunNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HunNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 4*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(10*funit, 10*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(10*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(10*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(10*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(10*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(10*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(16*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(16*funit, 16*funit, 3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(16*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(32*funit, 32*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(32*funit, 32*funit, 3, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.dec1to3std = nn.Sequential(
            # in: 32
            # out: 32
            nn.ConvTranspose2d(32*funit, 32*funit, 3, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.dec3to6str = nn.Sequential(
            # in: 32+32
            # out: 32
            nn.ConvTranspose2d(64*funit, 32*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )

        self.dec6to10std = nn.Sequential(
            # in: 32+32, out: 8
            nn.ConvTranspose2d(64*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 32+32, out: 8
            nn.ConvTranspose2d(64*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.dec10to14std = nn.Sequential(
            # in: 16+16, out: 8
            nn.ConvTranspose2d(32*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 16+16, out:8
            nn.ConvTranspose2d(32*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.dec14to42str = nn.Sequential(
            # in: 16+16, out:16
            nn.ConvTranspose2d(32*funit, 16*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.dec42to46std = nn.Sequential(
            # in: 16+16, out: 5
            nn.ConvTranspose2d(32*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 16+16, out:5
            nn.ConvTranspose2d(32*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec46to50std = nn.Sequential(
            # in: 10+10, out: 5
            nn.ConvTranspose2d(20*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 10+10, out: 5
            nn.ConvTranspose2d(20*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec50to150str = nn.Sequential(
            # in: 10+10, out: 10
            nn.ConvTranspose2d(20*funit, 10*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(10*funit),
            nn.ReLU(inplace=True),
        )
        self.dec150to154std = nn.Sequential(
            # in: 10+10, out: 3
            nn.ConvTranspose2d(20*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 10+10, out: 3
            nn.ConvTranspose2d(20*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+6, out: 2
            nn.ConvTranspose2d(12*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+6, out: 2
            nn.ConvTranspose2d(12*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+4
            nn.ConvTranspose2d(8*funit, 3, 3),
            nn.Sigmoid(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        upc158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        upc154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        upc50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        upc46 = cat46.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        upc14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        upc10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)

#160
class HuNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HuNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        upc158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        upc154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        upc50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        upc46 = cat46.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        upc14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        upc10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)
#160
class HuDisc(nn.Module):
    def __init__(self):
        funit = 32
        super(HuDisc, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        return self.enc3to1std(str3)

#144
class Hul144Disc(nn.Module):
    def __init__(self, input_channels = 3, funit = 32, finalpool = False):
        super(Hul144Disc, self).__init__()
        self.funit = funit
        self.enc144to142std = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc142to138std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc138to134std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc142to138dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc138to134dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc144to134dil = nn.Sequential(
            nn.Conv2d(input_channels, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc134to132std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc132to44str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc44to40std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc40to36std = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )

        self.enc44to40dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc40to36dil = nn.Sequential(
            nn.Conv2d(6*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc36to12str = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )

        self.enc12to8std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc8to4std = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )

        self.enc12to8dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc8to4dil = nn.Sequential(
            nn.Conv2d(12*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        if not finalpool:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 4*funit, 3, bias=True),
                nn.PReLU(init=0.01),
            )
            self.decide = nn.Sequential(
                nn.Conv2d(4*funit, 1, 2),
                nn.Sigmoid()
            )
        else:
            self.enc4to2std = nn.Sequential(
                nn.Conv2d(12*funit, 1, 3, bias=True),
                nn.Sigmoid(),
            )
            self.decide = nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
            )
    def forward(self, x):
        lay_i = self.enc144to142std(x)
        lay_i = torch.cat([self.enc142to138std(lay_i), self.enc142to138dil(lay_i)], 1)
        layer = torch.cat([self.enc138to134std(lay_i), self.enc138to134dil(lay_i), self.enc144to134dil(x)], 1)
        del(x, lay_i)
        layer = self.enc134to132std(layer)
        layer = self.enc132to44str(layer)
        layer = torch.cat([self.enc44to40std(layer), self.enc44to40dil(layer)], 1)
        layer = torch.cat([self.enc40to36std(layer), self.enc40to36dil(layer)], 1)
        layer = self.enc36to12str(layer)
        layer = torch.cat([self.enc12to8std(layer), self.enc12to8dil(layer)], 1)
        layer = torch.cat([self.enc8to4std(layer), self.enc8to4dil(layer)], 1)
        layer = self.enc4to2std(layer)
        return self.decide(layer)

# single-net: BS = 17 w/ 8GB GPU, 27 w/ 11GB GPU
# PReLU w/BN 128-to-128 generator
class Hul128Net(nn.Module):
    def __init__(self):
        funit = 32
        super(Hul128Net, self).__init__()
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
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc126to122dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc122to118dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc128to118dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc118to114std = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc118to114dil = nn.Sequential(
            nn.Conv2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc114to38str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.enc38to34std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc38to34dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc34to30dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc30to10str = nn.Sequential(
            nn.Conv2d(4*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )

        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2std = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
            nn.Conv2d(6*funit, 6*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc6to2dil = nn.Sequential(
            nn.Conv2d(6*funit, 6*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(6*funit),
        )
        self.dec2to6std = nn.Sequential(
            # in 0+12 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec2to6dil = nn.Sequential(
            # in: 0+8 out 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec6to10dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec10to30str = nn.Sequential(
            # in: 4+6, out:5
            nn.ConvTranspose2d(10*funit, 5*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(5*funit),
        )
        self.dec30to34std = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec30to34dil = nn.Sequential(
            # in: 4+5, out:3
            nn.ConvTranspose2d(9*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38std = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec34to38dil = nn.Sequential(
            # in: 4+6, out: 3
            nn.ConvTranspose2d(10*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec38to114str = nn.Sequential(
            # in: 4+6, out: 4
            nn.ConvTranspose2d(10*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.dec114to118std = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec114to118dil = nn.Sequential(
            # in: 4+4, out: 3
            nn.ConvTranspose2d(8*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122std = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec118to122dil = nn.Sequential(
            # in: 6+6, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
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
            nn.ConvTranspose2d(8*funit, 3, 3),
            nn.ReLU(),
        )

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
        return self.dec126to128std(l126)


# 160-160 PReLU w/BN generator
class Hul160Net(nn.Module):
    def __init__(self):
        funit = 32
        super(Hul160Net, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            nn.PReLU(init=0.01),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.PReLU(init=0.01),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(2*funit),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(4*funit),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
            nn.BatchNorm2d(3*funit),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            nn.PReLU(init=0.01),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # down
        # 160 to 150
        l158 = self.enc160to158std(x)
        l154 = torch.cat([self.enc158to154std(l158), self.enc158to154dil(l158)], 1)
        l150 = torch.cat([self.enc154to150std(l154), self.enc154to150dil(l154), self.enc160to150dil(x)], 1)
        del(x)
        l50 = self.enc150to50str(l150)
        l46 = torch.cat([self.enc50to46std(l50), self.enc50to46dil(l50)], 1)
        l42 = torch.cat([self.enc46to42std(l46), self.enc46to42dil(l46)], 1)
        l14 = self.enc42to14str(l42)
        l10 = torch.cat([self.enc14to10std(l14), self.enc14to10dil(l14)], 1)
        l6 = torch.cat([self.enc10to6std(l10), self.enc10to6dil(l10)], 1)
        l3 = self.enc6to3str(l6)    # k2s2
        l1 = self.enc3to1std(l3)
        # up
        l3 = torch.cat([l3, self.dec1to3std(l1)], 1)
        del(l1)

        l6 = torch.cat([l6, self.dec3to6str(l3)], 1)
        del(l3)
        l10 = torch.cat([l10, self.dec6to10std(l6), self.dec6to10dil(l6)], 1)
        del(l6)
        l14 = torch.cat([l14, self.dec10to14std(l10), self.dec10to14dil(l10)], 1)
        del(l10)
        l42 = torch.cat([l42, self.dec14to42str(l14)], 1)
        l46 = torch.cat([l46, self.dec42to46std(l42), self.dec42to46dil(l42)], 1)
        del(l42)
        l50 = torch.cat([l50, self.dec46to50std(l46), self.dec46to50dil(l46)], 1)
        del(l46)
        l150 = torch.cat([l150, self.dec50to150str(l50)], 1)
        del(l50)
        l154 = torch.cat([l154, self.dec150to154std(l150), self.dec150to154dil(l150)], 1)
        del(l150)
        l158 = torch.cat([l158, self.dec154to158std(l154), self.dec154to158dil(l154)], 1)
        del(l154)
        return self.dec158to160std(l158)

HulNet = Hul160Net  # compatibility

class HunkyNet(nn.Module):
    # possible input size: 224+int*16
    def __init__(self):
        super(HunkyNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc3 = nn.Sequential(
            nn.Conv2d(96,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc4 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc5 = nn.Sequential(
        nn.Conv2d(256,512,3),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,3),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        )
        # down
        self.encdec = nn.Sequential(
            nn.Conv2d(512,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.dec2 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec3 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec4 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(128,96,2,stride=2)
        self.dec5 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.ConvTranspose2d(96,64,2,stride=2)
        self.dec6 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3,5),
            nn.Sigmoid(),
        )
    def forward(self, x):
        residuals = []
        layer = self.enc1(x)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc2(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc3(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc4(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc5(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.encdec(layer)
        layer = self.up1(layer)
        layer = layer+residuals.pop()
        layer = self.dec2(layer)
        layer = self.up2(layer)
        layer = layer+residuals.pop()
        layer = self.dec3(layer)
        layer = self.up3(layer)
        layer = layer+residuals.pop()
        layer = self.dec4(layer)
        layer = self.up4(layer)
        layer = layer+residuals.pop()
        layer = self.dec5(layer)
        layer = self.up5(layer)
        layer = layer+residuals.pop()
        layer = self.dec6(layer)
        return layer

