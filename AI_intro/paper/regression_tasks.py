import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import GoogLeNet_Weights, Inception_V3_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights, DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, MobileNet_V2_Weights, \
    MNASNet1_0_Weights

from LogME import LogME


pretrained_models = ["googlenet", "inception_v3", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "mobilenet_v2", "mnasnet1_0"]

data_path = "E:/DataSets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")