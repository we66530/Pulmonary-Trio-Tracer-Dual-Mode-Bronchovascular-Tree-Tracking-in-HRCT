import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_f, out_f, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_f), nn.ReLU(True),
            nn.Conv2d(out_f, out_f, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_f)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_f)
            )
        self.relu = nn.ReLU(True)
    def forward(self, x): 
        return self.relu(self.conv(x) + self.shortcut(x))

class TrioContextUNet(nn.Module):
    """
    3-Channel Input (2.5D: z-1, z, z+1) 
    3-Channel Output (Artery, Vein, Bronchi)
    """
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(BasicResBlock(3, 64), BasicResBlock(64, 64))
        self.enc2 = nn.Sequential(BasicResBlock(64, 128), BasicResBlock(128, 128))
        self.enc3 = nn.Sequential(BasicResBlock(128, 256), BasicResBlock(256, 256))
        self.bottleneck = nn.Sequential(BasicResBlock(256, 512), BasicResBlock(512, 512))
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = BasicResBlock(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = BasicResBlock(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = BasicResBlock(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
    

class BronchiExpertUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f), nn.ReLU(True),
                nn.Conv2d(out_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f), nn.ReLU(True)
            )
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, 1) # 1-Channel Output

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2)); b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)