import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LONG(nn.Module):
    def __init__(self):
        super(LONG, self).__init__()
        my_momentum = 0.01
        # CBS
        # (32x3x6x6)  0 32 3 622
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6, padding=2, stride=2, bias=True)
        # self.bn1 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu0 = nn.SiLU(inplace=True)
        # CBS
        # (64x32x3x3) 3 64 32 312
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=True)
        # self.bn2 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu3 = nn.SiLU(inplace=True)
        # SCP1_1
        # (32x64x1x1) 6 32 64 101
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, stride=1, bias=True)
        # self.bn3 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu6 = nn.SiLU(inplace=True)
        # (32x32x1x1)
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1, bias=True)
        # self.bn3 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu9 = nn.SiLU(inplace=True)
        # (32x32x3x3)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=True)
        # self.bn3 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu12 = nn.SiLU(inplace=True)
        # (32x64x1x1)
        self.conv16 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, stride=1,bias=True)
        # self.bn3 = nn.BatchNorm2d(32, momentum=my_momentum)
        self.relu16 = nn.SiLU(inplace=True)
        # CBS
        # 64x64
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu20 = nn.SiLU(inplace=True)

        # (128x64)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True)
        self.relu23 = nn.SiLU(inplace=True)
        # 64x128
        self.conv26 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu26 = nn.SiLU(inplace=True)
        # 64x64
        self.conv29 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu29 = nn.SiLU(inplace=True)
        # 64x64
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu32 = nn.SiLU(inplace=True)
        # 64x64
        self.conv36 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu36 = nn.SiLU(inplace=True)
        # 64x64
        self.conv39 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu39 = nn.SiLU(inplace=True)
        # 64x128
        self.conv43 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu43 = nn.SiLU(inplace=True)
        # 128x128
        self.conv47 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu47 = nn.SiLU(inplace=True)
        # 256x128
        self.conv50 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=True)
        self.relu50 = nn.SiLU(inplace=True)
        # 128x256
        self.conv53 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu53 = nn.SiLU(inplace=True)
        # 128x128
        self.conv56 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu56 = nn.SiLU(inplace=True)
        # 128x128
        self.conv59 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu59 = nn.SiLU(inplace=True)
        # 128x128
        self.conv63 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu63 = nn.SiLU(inplace=True)
        # 128x128
        self.conv66 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu66 = nn.SiLU(inplace=True)
        # 128x128
        self.conv70 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu70 = nn.SiLU(inplace=True)
        # 128 128
        self.conv73 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu73 = nn.SiLU(inplace=True)
        # 128 256
        self.conv77 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu77 = nn.SiLU(inplace=True)
        # 256 256
        self.conv81 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu81 = nn.SiLU(inplace=True)
        # 512 256
        self.conv84 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, bias=True)
        self.relu84 = nn.SiLU(inplace=True)
        # 256 512 101
        self.conv87 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu87 = nn.SiLU(inplace=True)
        # 90 256 256 101 依次为编号  out_channels, in_channels, kernel_size, padding, stride
        self.conv90 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu90 = nn.SiLU(inplace=True)
        # 93 256 256 311
        self.conv93 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu93 = nn.SiLU(inplace=True)
        # 97 256 512 101
        self.conv97 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu97 = nn.SiLU(inplace=True)
        # 101 512 512 101
        self.conv101 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu101 = nn.SiLU(inplace=True)
        # 104 256 512 101
        self.conv104 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu104 = nn.SiLU(inplace=True)
        #
        # MaxPool
        # 107 521
        self.maxpooling107 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        # 108 521
        self.maxpooling108 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        # 109 521
        self.maxpooling109 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)

        # 111 512 1024 101
        self.conv111 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu111 = nn.SiLU(inplace=True)

        # 114 256 512 101
        self.conv114 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu114 = nn.SiLU(inplace=True)
        # 120 128 512 101
        self.conv120 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu120 = nn.SiLU(inplace=True)
        # 123 128 128 101
        self.conv123 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu123 = nn.SiLU(inplace=True)
        # 126 128 128 311
        self.conv126 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu126 = nn.SiLU(inplace=True)
        # 129 128 512 101
        self.conv129 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu129 = nn.SiLU(inplace=True)
        # 133 256 256 101
        self.conv133 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu133 = nn.SiLU(inplace=True)
        # 136 128 256 101
        self.conv136 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu136 = nn.SiLU(inplace=True)
        # 142 64 256 101
        self.conv142 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu142 = nn.SiLU(inplace=True)
        # 145 64 64 101
        self.conv145 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu145 = nn.SiLU(inplace=True)
        # 148 64 64 311
        self.conv148 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu148 = nn.SiLU(inplace=True)
        # 151 64 256 101
        self.conv151 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu151 = nn.SiLU(inplace=True)
        # 155 128 128 101
        self.conv155 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu155 = nn.SiLU(inplace=True)
        # 158 128 128 312
        self.conv158 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True)
        self.relu158 = nn.SiLU(inplace=True)
        # 162 128 256 101
        self.conv162 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu162 = nn.SiLU(inplace=True)
        # 165 128 128 101
        self.conv165 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu165 = nn.SiLU(inplace=True)
        # 168 128 128 311
        self.conv168 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu168 = nn.SiLU(inplace=True)
        # 171 128 256 101
        self.conv171 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu171 = nn.SiLU(inplace=True)
        # 175 256 256 101
        self.conv175 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu175 = nn.SiLU(inplace=True)
        # 178 256 256 312
        self.conv178 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2, bias=True)
        self.relu178 = nn.SiLU(inplace=True)
        # 182 256 512 101
        self.conv182 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu182 = nn.SiLU(inplace=True)
        # 185 256 256 101
        self.conv185 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu185 = nn.SiLU(inplace=True)
        # 188 256 256 311
        self.conv188 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu188 = nn.SiLU(inplace=True)
        # 191 256 512 101
        self.conv191 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu191 = nn.SiLU(inplace=True)
        # 195 512 512 101
        self.conv195 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, bias=True)
        self.relu195 = nn.SiLU(inplace=True)

        # 198 255 128 101
        self.conv198 = nn.Conv2d(in_channels=128, out_channels=255, kernel_size=1, padding=0, stride=1, bias=True)
        # 217 255 256 101
        self.conv217 = nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, padding=0, stride=1, bias=True)
        # 236 255 512 101
        self.conv236 = nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, im_data):
        # CBS 3x640x640
        sig1 = self.conv0(im_data)
        mul2 = self.relu0(sig1)
        # CBS
        sig4 = self.conv3(mul2)
        mul5 = self.relu3(sig4)
        # CSP1-1
        sig7 = self.con6(mul5)
        mul8 = self.relu6(sig7)

        sig10 = self.con9(mul8)
        mul11 = self.relu9(sig10)

        sig13 = self.con12(mul11)
        mul14 = self.relu12(sig13)
        add15 = mul14 + mul8
        # 旁路卷积
        sig17 = self.conv16(mul5)
        mul18 = self.relu16(sig17)
        cat19 = torch.cat([add15, mul18],1)
        #
        sig21 = self.conv20(cat19)
        mul22 = self.relu20(sig21)

        # CBS
        sig24 = self.conv23(mul22)
        mul25 = self.relu23(sig24)   # 此处结果传递到Neck层
        # CSP1_2 Resc*2
        sig27 = self.conv26(mul25)
        mul28 = self.relu26(sig27)  # //

        sig30 = self.conv29(mul28)
        mul31 = self.relu29(sig30)

        sig30 = self.conv32(mul31)
        mul34 = self.relu32(sig30)

        # Add35
        add35 = mul34 + mul28  # //

        sig37 = self.conv36(add35)
        mul38 = self.relu36(sig37)

        sig40 = self.conv39(mul38)
        mul41 = self.relu39(sig40)

        add42 = mul41 + add35
        # Conv
        sig44 = self.conv43(mul25)
        mul45 = self.relu43(sig44)  # /

        cat46 = torch.cat([add42, mul45], 1)
        sig48 = self.conv47(cat46)
        mul49 = self.relu47(sig48)   # ///.

        sig51 = self.conv50(mul49)
        mul52 = self.relu50(sig51)  # //.

        sig54 = self.conv53(mul52)
        mul55 = self.relu53(sig54)  # /

        sig57 = self.conv56(mul55)
        mul58 = self.relu56(sig57)

        sig60 = self.conv59(mul58)
        mul61 = self.relu59(sig60)
        add62 = mul61 + mul55  # /
        sig64 = self.conv63(add62)
        mul65 = self.relu63(sig64)
        sig67 = self.conv66(mul65)
        mul68 = self.relu66(sig67)
        add69 = mul68 + add62   # /
        sig71 = self.conv70(add69)
        mul72 = self.relu70(sig71)
        sig74 = self.conv73(mul72)
        mul75 = self.relu73(sig74)

        add76 = mul75 + add69  # /.


















