import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import GoogLeNet_Weights, Inception_V3_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights, DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, MobileNet_V2_Weights, \
    MNASNet1_0_Weights

from LogME import LogME

# googlenet == inception_v1; mnasnet1_0 == NASNet-A Mobile
pretrained_models = ["googlenet", "inception_v3", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "mobilenet_v2", "mnasnet1_0"]
dataset_names = ["Aircraft", "Caltech", "Cars", "CIFAR10", "CIFAR100", "DTD", "Pets", "SUN"]

data_path = "E:/DataSets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 48


def grayscale_to_rgb(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def main(dataset_name: str):
    scores = {}
    image_size = 224
    for model in pretrained_models:
        if model == "inception_v3":
            image_size = 299
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(dataset_name, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        scores[model] = get_score(model, data_loader)
        print(f"{model}: {scores[model]}")
    print(sorted(scores, key=lambda x: x[1][0]))


def get_score(model, data_loader):
    print(f"Evaluating {model}...")
    net = get_net(model).to(device)
    if model in pretrained_models[0:5]:
        fc_layer = net.fc
    elif model in pretrained_models[5:8]:
        fc_layer = net.classifier
    else:
        fc_layer = net.classifier[-1]
    F, Y, accuracy = forward(data_loader, net, fc_layer)

    logme = LogME(is_regression=False)
    return logme.fit(F.numpy(), Y.numpy()), accuracy


def get_net(model):
    if model == "googlenet":
        return models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    if model == "inception_v3":
        return models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    if model == "resnet50":
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    if model == "resnet101":
        return models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    if model == "resnet152":
        return models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    if model == "densenet121":
        return models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    if model == "densenet169":
        return models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
    if model == "densenet201":
        return models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    if model == "mobilenet_v2":
        return models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    if model == "mnasnet1_0":
        return models.mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)


def get_dataset(dataset_name: str, transform):
    if dataset_name == "Aircraft":
        return datasets.FGVCAircraft(root=data_path + dataset_name, download=True, transform=transform)
    if dataset_name == "Caltech":
        return datasets.Caltech101(root=data_path + dataset_name, download=True, transform=transform)
    if dataset_name == "Cars":
        return datasets.StanfordCars(root=data_path + dataset_name, download=False, transform=transform, split="test")  # automatic download is not working
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=data_path + dataset_name, download=True, transform=transform, train=False)
    if dataset_name == "CIFAR100":
        return datasets.CIFAR100(root=data_path + dataset_name, download=True, transform=transform, train=False)
    if dataset_name == "DTD":
        return datasets.DTD(root=data_path + dataset_name, download=True, transform=transform, split="val")
    if dataset_name == "Pets":
        return datasets.OxfordIIITPet(root=data_path + dataset_name, download=True, transform=transform)
    if dataset_name == "SUN":
        return datasets.SUN397(root=data_path + dataset_name, download=True, transform=transform)


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
    for dataset_name in dataset_names[2:]:
        main(dataset_name)
        print("-" * 50)