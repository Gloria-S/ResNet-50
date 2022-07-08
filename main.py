import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# from restnet18.restnet18 import RestNet18
from resnet50 import ResNet50
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 定义钩子函数，获取指定层名称的特征
activation = {} # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 原文链接：https://blog.csdn.net/weixin_45826022/article/details/118830531


#  用CIFAR-10 数据集进行实验

def main():
    batchsz = 128

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()

        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)

            # 从测试集中读取一张图片，并显示出来
            img_path = 'E:/学习资料/大学/AI实验/dataset/cifar/test/607_cat.png'
            img = Image.open(img_path)
            imgarray = np.array(img) / 255.0

            plt.figure(figsize=(8, 8))
            plt.imshow(imgarray)
            plt.axis('off')
            # plt.show()

            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            input_img = transform(img).unsqueeze(0)
            print(input_img.shape)

            # 获取layer1里面的bn3层的结果，浅层特征
            model.layer1[1].register_forward_hook(get_activation('bn3'))  # 为layer1中第2个模块的bn3注册钩子
            _ = model(input_img.cuda())

            bn3 = activation['bn3']  # 结果将保存在activation字典中
            print(bn3.shape)

            # 可视化结果，显示前64张
            plt.figure(figsize=(12, 12))
            for i in range(64):
                plt.subplot(8, 8, i + 1)
                plt.imshow(bn3[0, i, :, :].cpu(), cmap='gray')
                plt.axis('off')
            # plt.show()
            plt.savefig(str(epoch)+".png")


if __name__ == '__main__':
    main()

# 原文链接：https://blog.csdn.net/weixin_36979214/article/details/108883541