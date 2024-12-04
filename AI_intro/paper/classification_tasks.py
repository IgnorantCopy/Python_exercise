import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from LogME import LogME

# googlenet == inception_v1; mnasnet1_0 == NASNet-A Mobile
pretrained_models = ["googlenet", "inception_v3", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "mobilenet_v2", "mnasnet1_0"]
dataset_names = ["Aircraft", "Birdsnap", "Caltech", "Cars", "CIFAR10", "CIFAR100", "DTD", "Pets", "SUN"]

data_path = "E:/DataSets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 48


def main(dataset_name: str):
    scores = {}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for model in pretrained_models:
        if model == "inception_v3":
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        dataset = get_dataset(dataset_name, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        scores[model] = get_score(model, data_loader)
    print(sorted(scores, key=lambda x: x[1]))


def get_score(model, data_loader):
    print(f"Evaluating {model}...")
    net = get_net(model)
    if model in models[0:4]:
        fc_layer = net.fc
    elif model in models[4:6]:
        fc_layer = net.classifier
    else:
        fc_layer = net.classifier[-1]
    F, Y = forward(data_loader, net, fc_layer)

    logme = LogME(is_regression=False)
    return logme.fit(F.numpy(), Y.numpy())


def get_net(model):
    if model == "googlenet":
        return models.googlenet(pretrained=True)
    if model == "inception_v3":
        return models.inception_v3(pretrained=True)
    if model == "resnet50":
        return models.resnet50(pretrained=True)
    if model == "resnet101":
        return models.resnet101(pretrained=True)
    if model == "resnet152":
        return models.resnet152(pretrained=True)
    if model == "densenet121":
        return models.densenet121(pretrained=True)
    if model == "densenet169":
        return models.densenet169(pretrained=True)
    if model == "densenet201":
        return models.densenet201(pretrained=True)
    if model == "mobilenet_v2":
        return models.mobilenet_v2(pretrained=True)
    if models == "mnasnet1_0":
        return models.mnasnet1_0(pretrained=True)


def get_dataset(dataset_name, transform):
    if dataset_name == "Aircraft":
        return datasets.FGVCAircraft(root=data_path + dataset_name, download=True, transform=transform)
    if dataset_name == "Birdsnap":
        pass
    if dataset_name == "Caltech":
        return datasets.Caltech101(root=data_path + dataset_name, download=True, transform=transform)
    if dataset_name == "Cars":
        return datasets.StanfordCars(root=data_path + dataset_name, download=True, transform=transform, split="test")
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

    def hook_fn(cin):
        F.append(cin[0].detach().to(device))
    forward_hook = fc_layer.register_forward_hook(hook_fn)

    net.eval()
    with torch.no_grad():
        for _, (data, ground_truth) in enumerate(data_loader):
            Y.append(ground_truth.to(device))
            data = data.to(device)
            net(data)
    forward_hook.remove()
    F = torch.cat([i for i in F])
    Y = torch.cat([i for i in Y])
    return F, Y


if __name__ == '__main__':
    for dataset_name in dataset_names:
        main(dataset_name)
        print("-" * 50)