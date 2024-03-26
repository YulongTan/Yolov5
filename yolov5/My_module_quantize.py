# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from torch.autograd import Variable
# import numpy as np
#
# torch.set_default_dtype(torch.float64)
#
# def quantize(x, num_bits, fl_bits):
#     FL = fl_bits
#     IL = num_bits - fl_bits
#     # MIN = -(1 << (IL - 1))
#     MIN = -2 ** (IL - 1)
#     MAX = -MIN - 2 ** (-FL)
#     q = torch.floor((x * (2 ** FL))) / 2 ** FL   # torch.floor
#     q_copy = q
#     # q = torch.clip(q, MIN, MAX)
#     q = torch.clamp(q, MIN, MAX)
#     # print(torch.max(torch.abs(q_copy - q)))
#     return q
#     # import torch
#     # import torch.nn as nn
#     # import numpy as np
#     #
#     # # 加载保存在.npy文件中的卷积核权重（假设已经是四维形式）
#     # kernel_weights = np.load('conv_kernel.npy')
#     #
#     # # 将NumPy数组转换为PyTorch张量
#     # kernel_weights = torch.tensor(kernel_weights, dtype=torch.float32)
#     #
#     # # 创建 nn.Conv2d 模块，指定输入通道数、输出通道数和卷积核大小
#     # conv_layer = nn.Conv2d(in_channels=kernel_weights.shape[1], out_channels=kernel_weights.shape[0],
#     #                        kernel_size=(kernel_weights.shape[2], kernel_weights.shape[3]), bias=True)
#     #
#     # # 将加载的卷积核权重赋值给卷积层
#     # conv_layer.weight.data = kernel_weights
#     #
#     # # 将偏置设置为1
#     # conv_layer.bias.data.fill_(1)
#     #
#     # # 创建输入数据
#     # input_data = torch.randn(1, kernel_weights.shape[1], 5, 5)  # 输入数据形状为[N, C, H, W]
#     #
#     # # 进行卷积操作
#     # output = conv_layer(input_data)
#     #
#     # print(output)

# import torch
# # 加载.pt文件
# # model = torch.load('D:/LoongArch/LONG/my_model.pt')['model'].float().fuse().eval()
#
# model = torch.load('D:/LoongArch/LONG/yolov5s.onnx')['model'].float().fuse().eval()
#
# # 打开文件并写入值
# with open('D:/LoongArch/LONG/weights/weights.txt', 'w') as file:
#     # file.write(str(model))  # 将模型结构打印到文件中
#
#     for name, param in model.named_parameters():
#         file.write(f"Layer: {name} - Size: {param.size()}\n")
#         file.write(str(param) + "\n\n")  # 将每一层的权重参数打印到文件中
#
# import torch
#
# # 加载模型文件
# model_file = torch.load('D:/LoongArch/LONG/my_model.pt')
#
# # 查看模型文件中的所有键
# keys = model_file.keys()
# print(keys)

# import torch
# import onnx
# from onnx import numpy_helper
#
# # 加载 ONNX 模型
# onnx_model = onnx.load('D:/LoongArch/LONG/yolov5s.onnx')
#
# # 打开文件并写入模型结构
# with open('D:/LoongArch/LONG/weights/yolov5s_structure.txt', 'w') as file:
#     file.write(str(onnx_model.graph))
#
# # 打开文件并写入权重值
# with open('D:/LoongArch/LONG/weights/yolov5s_weights.txt', 'w') as file:
#     for tensor in onnx_model.graph.initializer:
#         name = tensor.name
#         numpy_array = numpy_helper.to_array(tensor)
#         file.write(f"Name: {name}, Shape: {numpy_array.shape}\n")
#         file.write(str(numpy_array) + "\n\n")

# import onnx
#
# # 加载ONNX模型
# model_path = 'D:/LoongArch/LONG/my_model.onnx'
# model = onnx.load(model_path)
#
# # 获取模型的权重
# initializers = model.graph.initializer
#
# # 打印所有权重的键
# for initializer in initializers:
#     print(initializer.name)
# import onnx
# from onnx import numpy_helper
#
# # Step 1: 加载 YOLOv5 的 ONNX 模型，并获取权重参数
# yolov5_model_path = 'D:/LoongArch/LONG/yolov5s.onnx'
# yolov5_model = onnx.load(yolov5_model_path)
#
# # 获取 YOLOv5 的权重参数
# yolov5_initializers = yolov5_model.graph.initializer
# yolov5_weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in yolov5_initializers}
#
# # Step 2: 加载你自己的模型的 ONNX 文件
# my_model_path = 'D:/LoongArch/LONG/my_model.onnx'
# my_model = onnx.load(my_model_path)
#
# # 获取你自己模型的权重参数
# my_model_initializers = my_model.graph.initializer
#
# # 遍历你自己模型的权重参数，并根据层的位置将 YOLOv5 的权重映射到你自己的模型中
# for i, initializer in enumerate(my_model_initializers):
#     # 检查是否有相应的权重参数
#     if i < len(yolov5_initializers):
#         # 更新权重参数
#         initializer.CopyFrom(numpy_helper.from_array(yolov5_weights[yolov5_initializers[i].name]))
#
# # 注意：如果你的模型的层数多于 YOLOv5 模型，则需要额外处理剩余的权重参数
#
# # 注意：如果你的模型的层数少于 YOLOv5 模型，则可能需要按需加载 YOLOv5 的部分权重
#
# # 不一致层的权重参数需要额外处理
#
# # 更新后的模型现在已经包含了 YOLOv5 的权重
# #
# import onnx
# from onnx import numpy_helper
#
# # 加载 YOLOv5 的 ONNX 模型
# yolov5_model_path = 'D:/LoongArch/LONG/yolov5s.onnx'
# yolov5_model = onnx.load(yolov5_model_path)
#
# # 获取 YOLOv5 的初始化器（即权重参数）
# yolov5_initializers = yolov5_model.graph.initializer
#
# # 定义每个文件中的最大行数
# max_lines = 1000
#
# # 遍历初始化器并将每个权重写入单独的文本文件
# for i, initializer in enumerate(yolov5_initializers):
#     # 获取权重名称和权重值
#     weight_name = initializer.name
#     weight_array = numpy_helper.to_array(initializer)
#
#     # 确定文件名
#     if "weight" in weight_name:
#         output_file = f'D:/LoongArch/LONG/weights/weight_{i}.txt'
#     elif "bias" in weight_name:
#         output_file = f'D:/LoongArch/LONG/weights/bias_{i}.txt'
#     else:
#         continue  # 如果不是权重或偏置，则跳过
#
#     # 将权重值分割成较小的块，并分批写入文件
#     with open(output_file, 'w') as file:
#         file.write(f"Name: {weight_name}\n")
#         file.write(f"Shape: {weight_array.shape}\n")
#
#         # 计算需要分割的次数
#         num_splits = len(weight_array) // max_lines
#         if len(weight_array) % max_lines != 0:
#             num_splits += 1
#
#         # 分割权重值并分批写入文件
#         for j in range(num_splits):
#             start_idx = j * max_lines
#             end_idx = (j + 1) * max_lines
#             chunk = weight_array[start_idx:end_idx]
#             file.write(str(chunk))
#             file.write('\n')  # 添加换行符，确保每个块在单独的行中

# import torch
#
# # 加载.pt文件
# model = torch.load('D:/LoongArch/LONG/Infrared-Object-Detection-main/yolov5/yolov5s.pt')['model'].float().fuse().eval()
#
# # 打开文件并写入值
# with open('D:/LoongArch/LONG/weights/new_weight.txt', 'w') as file:
#     file.write(str(model))  # 将模型结构打印到文件中
#
#     for name, param in model.named_parameters():
#         file.write(f"Layer: {name} - Size: {param.size()}\n")
#         file.write(str(param) + "\n\n")   #  将每一层的权重参数打印到文件中

# -----导入mul and add
import numpy as np

# 加载.npy文件
mul_add_path = 'D:/LoongArch/LONG/Infrared-Object-Detection-main/my_weights_add_mul'
tensor = np.load(mul_add_path + '/add_206.npy')

# 打印张量的形状
print("Tensor shape:", tensor.shape)
print(f'tesor_name: {tensor}')


