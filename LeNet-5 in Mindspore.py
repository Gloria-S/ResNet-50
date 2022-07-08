import os
import argparse
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1,status="train"):
    # 定义数据集
    cifar_ds = ds.Cifar10Dataset(data_path,usage=status)
    # 归一化
    rescale = 1.0 / 255.0
    # 平移
    shift = 0.0

    resize_op = CV.Resize((32, 32))
    rescale_op = CV.Rescale(rescale, shift)
    # 对于RGB三通道分别设定mean和std
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        # 随机裁剪
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        # 随机翻转
        random_horizontal_op = CV.RandomHorizontalFlip()
    # 通道变化
    channel_swap_op = CV.HWC2CHW()
    # 类型变化
    typecast_op = C.TypeCast(mstype.int32)

    # 算子运算
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op)
    if status == "train":
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_crop_op)
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_horizontal_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=resize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=rescale_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=normalize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=channel_swap_op)

    # shuffle
    cifar_ds = cifar_ds.shuffle(buffer_size=1000)
    # 切分数据集到batch_size
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    return cifar_ds

import mindspore.nn as nn
from mindspore.common.initializer import Normal

import numpy as np
from mindspore import Tensor
from PIL import Image
import matplotlib.pyplot as plt

def visualize(feature_map, loc, time_tuple):
    feature_map = feature_map.asnumpy()
    num_channel = feature_map.shape[0]
    if num_channel>=10: # 最多展示10层
        num_channel = 10
    for i in range(num_channel):
        feature_image = feature_map[i, :, :]
        feature_image-= feature_image.mean()
        feature_image/= feature_image.std ()
        feature_image*=  64
        feature_image+= 128
        pil_img = Image.fromarray(feature_image)
        plt.subplot(1, 10, i+1)
        plt.imshow(pil_img)
        plt.axis('off')
    plt.show()
    return 0

import time

class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        time_tuple = time.localtime(time.time())
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        visualize(x[0],0,time_tuple)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        visualize(x[0],1,time_tuple)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# class LeNet5(nn.Cell):
#     def __init__(self, num_class=10, num_channel=3):
#         super(LeNet5, self).__init__()
#         # 定义所需要的运算
#         self.conv1=nn.Conv2d(num_channel,64,3,pad_mode='valid')
#         self.conv2=nn.Conv2d(64,128,3,pad_mode='valid')
#         self.conv3=nn.Conv2d(128,128,3,pad_mode='valid')
#         self.fc1 = nn.Dense(512, 120, weight_init=Normal(0.02))
#         self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
#         self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
#         self.relu = nn.ReLU()
#         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()

#     def construct(self, x):
#         time_tuple = time.localtime(time.time())
#         # 使用定义好的运算构建前向网络
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool2d(x)
#         visualize(x[0],0,time_tuple)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.max_pool2d(x)
#         visualize(x[0],1,time_tuple)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.max_pool2d(x)
#         visualize(x[0],2,time_tuple)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# 实例化网络
net = LeNet5()

# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=1500, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(data_path, 32, repeat_size, status="train")
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)],dataset_sink_mode=sink_mode)
    
def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(data_path)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

context.set_context(mode=context.PYNATIVE_MODE)
train_epoch = 10
cifar_path = "./cifar-10-batches-bin"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, cifar_path, dataset_size, ckpoint, False)
test_net(model, cifar_path)

# from mindspore import load_checkpoint, load_param_into_net
# # 加载已经保存的用于测试的模型
# param_dict = load_checkpoint("checkpoint_lenet-09_1500.ckpt")
# # 加载参数到网络中
# load_param_into_net(net, param_dict)
