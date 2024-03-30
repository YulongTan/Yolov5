import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import numpy_helper
from torchvision import transforms
from PIL import ImageDraw
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.augmentations import letterbox

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5)
class LONG(nn.Module):
    def __init__(self):
        super(LONG, self).__init__()
        # my_momentum = 0.01
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
        self.maxpooling107 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1, dilation=1, ceil_mode=False)
        # 108 521
        self.maxpooling108 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1, dilation=1, ceil_mode=False)
        # 109 521
        self.maxpooling109 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1, dilation=1, ceil_mode=False)

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

        # 加载add and mul 的权重
        mul_add_path = 'D:/LoongArch/LONG/Infrared-Object-Detection-main/my_weights_add_mul'
        self.add206_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/add_206.npy')))
        self.add225_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/add_225.npy')))
        self.add244_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/add_244.npy')))
        self.mul214_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/mul_214.npy')))
        self.mul233_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/mul_233.npy')))
        self.mul252_weight = nn.Parameter(torch.from_numpy(np.load(mul_add_path + '/mul_252.npy')))
        # self.add206_weight = nn.Parameter(torch.randn(1, 3, 80, 80, 2))
        # self.add225_weight = nn.Parameter(torch.randn(1, 3, 40, 40, 2))
        # self.add244_weight = nn.Parameter(torch.randn(1, 3, 20, 20, 2))
        # self.mul214_weight = nn.Parameter(torch.randn(1, 3, 80, 80, 2))
        # self.mul233_weight = nn.Parameter(torch.randn(1, 3, 40, 40, 2))
        # self.mul252_weight = nn.Parameter(torch.randn(1, 3, 20, 20, 2))

    def load_weight(self, weight_path='D:/LoongArch/LONG/yolov5s.onnx'):
        yolov5_model = onnx.load(weight_path)
        # 获取 YOLOv5 的初始化器（即权重参数）
        yolov5_initializers = yolov5_model.graph.initializer

        # 遍历初始化器并将每个权重赋值给自己的模型
        for initializer in yolov5_initializers:
            # 获取权重名称和权重值
            weight_name = initializer.name
            weight_array = numpy_helper.to_array(initializer)
            # print(weight_name)
            if weight_name == 'model.0.conv.weight':
                self.conv0.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.0.conv.bias':
                self.conv0.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.1.conv.weight':
                self.conv3.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.1.conv.bias':
                self.conv3.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.2.cv1.conv.weight':
                self.conv6.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.2.cv1.conv.bias':
                self.conv6.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.2.m.0.cv1.conv.weight':
                self.conv9.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.2.m.0.cv1.conv.bias':
                self.conv9.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.2.m.0.cv2.conv.weight':
                self.conv12.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.2.m.0.cv2.conv.bias':
                self.conv12.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.2.cv2.conv.weight':
                self.conv16.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.2.cv2.conv.bias':
                self.conv16.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.2.cv3.conv.weight':
                self.conv20.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.2.cv3.conv.bias':
                self.conv20.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.3.conv.weight':
                self.conv23.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.3.conv.bias':
                self.conv23.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.cv1.conv.weight':
                self.conv26.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.cv1.conv.bias':
                self.conv26.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.m.0.cv1.conv.weight':
                self.conv29.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.m.0.cv1.conv.bias':
                self.conv29.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.m.0.cv2.conv.weight':
                self.conv32.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.m.0.cv2.conv.bias':
                self.conv32.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.m.1.cv1.conv.weight':
                self.conv36.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.m.1.cv1.conv.bias':
                self.conv36.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.m.1.cv2.conv.weight':
                self.conv39.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.m.1.cv2.conv.bias':
                self.conv39.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.cv2.conv.weight':
                self.conv43.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.cv2.conv.bias':
                self.conv43.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.4.cv3.conv.weight':
                self.conv47.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.4.cv3.conv.bias':
                self.conv47.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.5.conv.weight':
                self.conv50.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.5.conv.bias':
                self.conv50.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.cv1.conv.weight':
                self.conv53.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.cv1.conv.bias':
                self.conv53.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.0.cv1.conv.weight':
                self.conv56.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.0.cv1.conv.bias':
                self.conv56.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.0.cv2.conv.weight':
                self.conv59.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.0.cv2.conv.bias':
                self.conv59.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.1.cv1.conv.weight':
                self.conv63.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.1.cv1.conv.bias':
                self.conv63.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.1.cv2.conv.weight':
                self.conv66.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.1.cv2.conv.bias':
                self.conv66.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.2.cv1.conv.weight':
                self.conv70.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.2.cv1.conv.bias':
                self.conv70.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.m.2.cv2.conv.weight':
                self.conv73.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.m.2.cv2.conv.bias':
                self.conv73.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.cv2.conv.weight':
                self.conv77.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.cv2.conv.bias':
                self.conv77.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.6.cv3.conv.weight':
                self.conv81.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.6.cv3.conv.bias':
                self.conv81.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.7.conv.weight':
                self.conv84.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.7.conv.bias':
                self.conv84.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.8.cv1.conv.weight':
                self.conv87.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.8.cv1.conv.bias':
                self.conv87.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.8.m.0.cv1.conv.weight':
                self.conv90.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.8.m.0.cv1.conv.bias':
                self.conv90.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.8.m.0.cv2.conv.weight':
                self.conv93.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.8.m.0.cv2.conv.bias':
                self.conv93.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.8.cv2.conv.weight':
                self.conv97.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.8.cv2.conv.bias':
                self.conv97.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.8.cv3.conv.weight':
                self.conv101.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.8.cv3.conv.bias':
                self.conv101.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.9.cv1.conv.weight':
                self.conv104.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.9.cv1.conv.bias':
                self.conv104.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.9.cv2.conv.weight':
                self.conv111.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.9.cv2.conv.bias':
                self.conv111.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.10.conv.weight':
                self.conv114.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.10.conv.bias':
                self.conv114.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.13.cv1.conv.weight':
                self.conv120.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.13.cv1.conv.bias':
                self.conv120.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.13.m.0.cv1.conv.weight':
                self.conv123.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.13.m.0.cv1.conv.bias':
                self.conv123.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.13.m.0.cv2.conv.weight':
                self.conv126.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.13.m.0.cv2.conv.bias':
                self.conv126.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.13.cv2.conv.weight':
                self.conv129.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.13.cv2.conv.bias':
                self.conv129.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.13.cv3.conv.weight':
                self.conv133.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.13.cv3.conv.bias':
                self.conv133.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.14.conv.weight':
                self.conv136.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.14.conv.bias':
                self.conv136.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.17.cv1.conv.weight':
                self.conv142.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.17.cv1.conv.bias':
                self.conv142.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.17.m.0.cv1.conv.weight':
                self.conv145.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.17.m.0.cv1.conv.bias':
                self.conv145.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.17.m.0.cv2.conv.weight':
                self.conv148.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.17.m.0.cv2.conv.bias':
                self.conv148.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.17.cv2.conv.weight':
                self.conv151.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.17.cv2.conv.bias':
                self.conv151.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.17.cv3.conv.weight':
                self.conv155.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.17.cv3.conv.bias':
                self.conv155.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.18.conv.weight':
                self.conv158.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.18.conv.bias':
                self.conv158.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.20.cv1.conv.weight':
                self.conv162.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.20.cv1.conv.bias':
                self.conv162.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.20.m.0.cv1.conv.weight':
                self.conv165.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.20.m.0.cv1.conv.bias':
                self.conv165.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.20.m.0.cv2.conv.weight':
                self.conv168.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.20.m.0.cv2.conv.bias':
                self.conv168.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.20.cv2.conv.weight':
                self.conv171.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.20.cv2.conv.bias':
                self.conv171.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.20.cv3.conv.weight':
                self.conv175.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.20.cv3.conv.bias':
                self.conv175.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.21.conv.weight':
                self.conv178.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.21.conv.bias':
                self.conv178.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.23.cv1.conv.weight':
                self.conv182.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.23.cv1.conv.bias':
                self.conv182.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.23.m.0.cv1.conv.weight':
                self.conv185.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.23.m.0.cv1.conv.bias':
                self.conv185.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.23.m.0.cv2.conv.weight':
                self.conv188.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.23.m.0.cv2.conv.bias':
                self.conv188.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.23.cv2.conv.weight':
                self.conv191.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.23.cv2.conv.bias':
                self.conv191.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.23.cv3.conv.weight':
                self.conv195.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.23.cv3.conv.bias':
                self.conv195.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.24.m.0.weight':
                self.conv198.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.24.m.0.bias':
                self.conv198.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.24.m.1.weight':
                self.conv217.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.24.m.1.bias':
                self.conv217.bias = nn.Parameter(torch.tensor(weight_array))

            elif weight_name == 'model.24.m.2.weight':
                self.conv236.weight = nn.Parameter(torch.tensor(weight_array))
            elif weight_name == 'model.24.m.2.bias':
                self.conv236.bias = nn.Parameter(torch.tensor(weight_array))

    def forward(self, im_data):
        features = []

        def hook(module, input, output):
            features.append(output)

        target_conv_layer = self.conv136
        target_conv_layer.register_forward_hook(hook)

        # CBS 3x640x640
        sig1 = self.conv0(im_data)
        mul2 = self.relu0(sig1)
        # CBS
        sig4 = self.conv3(mul2)
        mul5 = self.relu3(sig4)
        # CSP1-1
        sig7 = self.conv6(mul5)
        mul8 = self.relu6(sig7)

        sig10 = self.conv9(mul8)
        mul11 = self.relu9(sig10)

        sig13 = self.conv12(mul11)
        mul14 = self.relu12(sig13)
        add15 = mul14 + mul8
        # 旁路卷积
        sig17 = self.conv16(mul5)
        mul18 = self.relu16(sig17)
        cat19 = torch.cat((add15, mul18), 1)
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
        mul49 = self.relu47(sig48)   # ///

        sig51 = self.conv50(mul49)
        mul52 = self.relu50(sig51)  # //

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

        sig78 = self.conv77(mul52)
        mul79 = self.relu77(sig78)

        cat80 = torch.cat([add76, mul79], 1)
        sig82 = self.conv81(cat80)
        mul83 = self.relu81(sig82)  # //
        sig85 = self.conv84(mul83)
        mul86 = self.relu84(sig85)  # //
        sig88 = self.conv87(mul86)
        mul89 = self.relu87(sig88)  # //
        sig91 = self.conv90(mul89)
        mul92 = self.relu90(sig91)
        sig94 = self.conv93(mul92)
        mul95 = self.relu93(sig94)

        add96 = mul89 + mul95

        sig98 = self.conv97(mul86)
        mul99 = self.relu97(sig98)
        cat100 = torch.cat([add96, mul99], 1)

        sig102 = self.conv101(cat100)
        mul103 = self.relu101(sig102)
        sig105 = self.conv104(mul103)
        mul106 = self.relu104(sig105)  #
        # max_pooling*3
        pool107 = self.maxpooling107(mul106)
        pool108 = self.maxpooling108(pool107)
        pool109 = self.maxpooling108(pool108)
        cat110 = torch.cat([mul106, pool107, pool108, pool109], 1)
        sig112 = self.conv111(cat110)
        mul113 = self.relu111(sig112)
        sig115 = self.conv114(mul113)
        mul116 = self.relu114(sig115)  # //

        resize118 = F.interpolate(mul116, scale_factor=2, mode='nearest')
        # cat119 = torch.cat([mul83, resize118], 1)      # /
        cat119 = torch.cat([resize118, mul83], 1)

        sig121 = self.conv120(cat119)
        mul122 = self.relu120(sig121)
        sig124 = self.conv123(mul122)
        mul125 = self.relu123(sig124)
        sig127 = self.conv126(mul125)
        mul128 = self.relu126(sig127)

        sig130 = self.conv129(cat119)
        mul131 = self.relu129(sig130)
        cat132 = torch.cat([mul128, mul131], 1)
        sig134 = self.conv133(cat132)
        mul135 = self.relu133(sig134)
        sig137 = self.conv136(mul135)
        mul138 = self.relu136(sig137)     # /

        resize140 = F.interpolate(mul138, scale_factor=2, mode='nearest')

        # cat141 = torch.cat([mul49, resize140], 1)   # //
        cat141 = torch.cat([resize140, mul49], 1)  # //

        sig143 = self.conv142(cat141)
        mul144 = self.relu142(sig143)
        sig146 = self.conv145(mul144)
        mul147 = self.relu145(sig146)
        sig149 = self.conv148(mul147)
        mul150 = self.relu148(sig149)

        sig152 = self.conv151(cat141)
        mul153 = self.relu151(sig152)
        cat154 = torch.cat([mul150, mul153], 1)
        sig156 = self.conv155(cat154)
        mul157 = self.relu155(sig156)   # //
        sig159 = self.conv158(mul157)
        mul160 = self.relu158(sig159)

        # cat161 = torch.cat([mul138, mul160], 1)  # //
        cat161 = torch.cat([mul160, mul138], 1)  # //

        sig163 = self.conv162(cat161)
        mul164 = self.relu162(sig163)
        sig166 = self.conv165(mul164)
        mul167 = self.relu165(sig166)
        sig169 = self.conv168(mul167)
        mul170 = self.relu168(sig169)

        sig172 = self.conv171(cat161)
        mul173 = self.relu171(sig172)
        cat174 = torch.cat([mul170, mul173], 1)  # //
        sig176 = self.conv175(cat174)
        mul177 = self.relu175(sig176)  # /
        sig179 = self.conv178(mul177)
        mul180 = self.relu178(sig179)
        # cat181 = torch.cat([mul116, mul180], 1)  # //
        cat181 = torch.cat([mul180, mul116], 1)

        sig183 = self.conv182(cat181)
        mul184 = self.relu182(sig183)
        sig186 = self.conv185(mul184)
        mul187 = self.relu185(sig186)
        sig189 = self.conv188(mul187)
        mul190 = self.relu188(sig189)

        sig192 = self.conv191(cat181)
        mul193 = self.relu191(sig192)
        cat194 = torch.cat([mul190, mul193], 1)

        sig196 = self.conv195(cat194)
        mul197 = self.relu195(sig196)

        # 最后三个分支

        cv198 = self.conv198(mul157)
        cv217 = self.conv217(mul177)
        cv236 = self.conv236(mul197)
        # Reshape
        reshape199 = cv198.reshape(1, 3, 85, 80, 80)
        reshape218 = cv217.reshape(1, 3, 85, 40, 40)
        reshape237 = cv236.reshape(1, 3, 85, 20, 20)
        # Transpose
        transpose200 = reshape199.permute(0, 1, 3, 4, 2)
        transpose219 = reshape218.permute(0, 1, 3, 4, 2)
        transpose238 = reshape237.permute(0, 1, 3, 4, 2)
        # sigmoid
        sig201 = torch.sigmoid(transpose200)
        sig220 = torch.sigmoid(transpose219)
        sig239 = torch.sigmoid(transpose238)
        # Split
        split202_2, split202_3, split202_1 = torch.split(sig201, [2, 2, 81], 4)
        split221_2, split221_3, split221_1 = torch.split(sig220, [2, 2, 81], 4)
        split240_2, split240_3, split240_1 = torch.split(sig239, [2, 2, 81], 4)

        mul204 = torch.mul(split202_2, 2)
        add206 = torch.add(mul204, self.add206_weight)
        mul208 = torch.mul(add206, 8)
        mul210 = torch.mul(split202_3, 2)
        pow212 = torch.pow(mul210, 2)
        mul214 = torch.mul(pow212, self.mul214_weight)
        # cat215 = torch.cat([split202_1, mul208, mul214], 4)
        cat215 = torch.cat([mul208, mul214, split202_1], 4)

        mul223 = torch.mul(split221_2, 2)
        add225 = torch.add(mul223, self.add225_weight)
        mul227 = torch.mul(add225, 16)
        mul229 = torch.mul(split221_3, 2)
        pow231 = torch.pow(mul229, 2)
        mul233 = torch.mul(pow231, self.mul233_weight)
        # cat234 = torch.cat([split221_1, mul227, mul233], 4)
        cat234 = torch.cat([mul227, mul233, split221_1], 4)

        mul242 = torch.mul(split240_2, 2)
        add244 = torch.add(mul242, self.add244_weight)
        mul246 = torch.mul(add244, 32)
        mul248 = torch.mul(split240_3, 2)
        pow250 = torch.pow(mul248, 2)
        mul252 = torch.mul(pow250, self.mul252_weight)
        # cat253 = torch.cat([split240_1, mul246, mul252], 4)
        cat253 = torch.cat([mul246, mul252, split240_1], 4)

        reshape216 = cat215.reshape(1, -1, 85)
        reshape235 = cat234.reshape(1, -1, 85)
        reshape253 = cat253.reshape(1, -1, 85)
        output = torch.cat([reshape216, reshape235, reshape253], 1)

        return output, features[-1]


def load_img(path, shape=(640, 640), stride=32):
    """
    get input image and make it acceptable to yolov5s
    Args:
        path: initial image path
        shape: standard input shape
        stride: shrink strides of image
    Returns: tensor which can input to network: (N,C,H,W)
    """
    input_img = cv2.imread(path)
    norm_img = letterbox(input_img, shape, stride=stride, auto=False)[0]
    norm_img = norm_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    np_img = np.ascontiguousarray(norm_img)
    tor_img = torch.from_numpy(np_img)
    tor_img = tor_img.float() / 255
    if len(tor_img.shape) == 3:
        tor_img = tor_img[None]  # expand for batch dim

    return tor_img


if __name__ == "__main__":
    model = LONG()
    # print(model)
    model.load_weight()
    model.eval()
    torch.save(model.state_dict(), "D:/LoongArch/LONG/my_model_final.pt")
    # # image_load
    # img_name = 'bus'
    # img_path = 'C:/Users/lenovo/Desktop/' + img_name + '.jpg'
    # tor_img = load_img(img_path)
    # print('input size:' + str(tor_img.shape))
    # pred = model.forward(tor_img)
    # print('output size:' + str(pred.shape))
    # # print(pred[0][0])
    # nms_pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
    # print(nms_pred)

    # image_load
    img_name = 'bus'
    # img_name = 'RC'
    # img_name = 'MY'
    img_path = 'C:/Users/lenovo/Desktop/' + img_name + '.jpg'
    tor_img = load_img(img_path)
    # 执行模型预测
    pred, feature_map = model.forward(tor_img)
    print(pred)
    # 执行非最大抑制
    # nms_pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
    nms_pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
    print(nms_pred)
    # 将张量转换回 PIL 图像对象
    tor_img_pil = transforms.ToPILImage()(tor_img[0].cpu()).convert("RGB")

    # 创建图像的副本，以便绘制边界框
    image_with_boxes_pil = tor_img_pil.copy()

    # 提取边界框信息并绘制边界框
    for nms_pred_item in nms_pred:
        for bbox_tensor in nms_pred_item:
            bbox = bbox_tensor.detach().cpu().numpy()  # 使用 detach() 创建不需要梯度的张量副本
            x1, y1, x2, y2 = bbox.astype(int)[:4]
            confidence = float(bbox[4])
            class_id = bbox.astype(int)[5]
            # 绘制边界框
            draw = ImageDraw.Draw(image_with_boxes_pil)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            # 显示类别和置信度
            label = f"Class: {int(class_id)}, Confidence: {confidence:.2f}"
            draw.text((x1, y1 - 10), label, fill=(0, 255, 0))
    #
    # num_channels_to_display = min(6, feature_map.shape[1])
    #
    # # 创建一个包含多个子图的 Matplotlib figure
    # fig, axes = plt.subplots(1, num_channels_to_display, figsize=(12, 4))
    #
    # for i in range(num_channels_to_display):
    #     feature_map_channel = feature_map[0, i].detach().numpy()  # 获取第 i 个通道的特征图
    #     # 对特征图数据进行归一化，确保在 [0, 1] 的范围内
    #     feature_map_normalized = (feature_map_channel - feature_map_channel.min()) / (
    #                 feature_map_channel.max() - feature_map_channel.min())
    #     # 在子图中显示特征图
    #     axes[i].imshow(feature_map_normalized, cmap='gray')
    #     axes[i].set_title(f'Channel {i + 1}')
    #
    # plt.tight_layout()
    # plt.show()

    # 显示带有边界框的图片
    image_with_boxes_pil.show()

    cv2.waitKey(0)  # 添加一个适当的延迟，直到按下任意键关闭窗口
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
    # print(model.add206_weight)

    # # --------------------将自己的模型导出为.onnx--------------------------------------------
    # torch.onnx.export(model,
    #                   torch.randn(1, 3, 640, 640),
    #                   "D:/LoongArch/LONG/my_model_final.onnx",
    #                   export_params=True,
    #                   opset_version=11,
    #                   do_constant_folding=True,
    #                   input_names=['input'],
    #                   output_names=['output'],
    #                   )
    # -----------------------------------------------------------------------------------

    # ----------------------打印某卷积层的卷积核和bias--------------------------------------
    # conv_layyer = getattr(model, 'conv0')
    # weights = conv_layyer.weight.data
    # bias = conv_layyer.bias.data
    # print("Convolution Layer Weight:")
    # print(weights)
    # print("Convolution Layer Bias:")
    # print(bias)
    # -----------------------------------------------------------------------------------






















