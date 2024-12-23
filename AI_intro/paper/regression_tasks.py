import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import GoogLeNet_Weights, Inception_V3_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights, DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, MobileNet_V2_Weights, \
    MNASNet1_0_Weights

from LogME import LogME

pretrained_models = ["googlenet", "inception_v3", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169",
                     "densenet201", "mobilenet_v2", "mnasnet1_0"]

data_path = "E:/DataSets/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    scores = {}
    # 图像的预处理，包括调整图像大小和归一化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到224x224大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.5], [0.5])  # 用于回归的标准化（灰度图像）
    ])
    for model_name in pretrained_models:
        if model_name == "inception_v3":
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        # 创建dSprites数据集的实例
        dataset = dSpritesDataset(data_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=48, shuffle=True, num_workers=4, pin_memory=True)
        scores[model_name] = get_score(model_name, data_loader)
        print(f"{model_name}: {scores[model_name]}")
    print(sorted(scores, key=lambda x: x[1]))


# 自定义数据集加载
class dSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # 加载数据（假设数据是图像和标签）
        # 数据集包含对应的标签 scale, orientation, x, y
        # 这里我们假设数据集已经是一个 numpy 数组或者类似结构
        self.images, self.labels, _, _ = np.load(data_path)  # 假设图像存储在该文件中

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换为PIL图像并应用变换
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)

        return image, label  # 返回图像和对应的回归标签（scale, orientation, x, y）


def get_score(model_name, data_loader):
    print(f"Evaluating {model_name}...")
    net = get_net(model_name).to(device)
    if model_name in pretrained_models[0:5]:
        fc_layer = net.fc
    elif model_name in pretrained_models[5:8]:
        fc_layer = net.classifier
    else:
        fc_layer = net.classifier[-1]
    F, Y, accuracy = forward(data_loader, net, fc_layer)

    logme = LogME(is_regression=True)
    return logme.fit(F.numpy(), Y.numpy()), accuracy


def get_net(model_name):
    if model_name == "googlenet":
        model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    if model_name == "inception_v3":
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    if model_name == "resnet101":
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    if model_name == "resnet152":
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    if model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    if model_name == "densenet169":
        model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
    if model_name == "densenet201":
        model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model_name.fc.in_features, 4)
    return model


def forward(data_loader, net, fc_layer):
    F = []
    Y = []

    def hook_fn(_, input, __):
        F.append(input[0].detach().to(device))

    forward_hook = fc_layer.register_forward_hook(hook_fn)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, ground_truth in data_loader:
            data, ground_truth = data.to(device), ground_truth.to(device)
            Y.append(ground_truth)
            output = net(data)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == ground_truth).sum().item()
    forward_hook.remove()
    F = torch.cat([i for i in F])
    Y = torch.cat([i for i in Y])
    return F, Y, 100 * correct / total


if __name__ == '__main__':
    main()
    print("-" * 50)
