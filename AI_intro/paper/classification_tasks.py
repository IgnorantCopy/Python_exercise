import os.path
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
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
lr = 1e-3
epoches = 100

def grayscale_to_rgb(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def main(dataset_name: str):
    scores = {}
    fine_tune(dataset_name)
    for model in pretrained_models:
        transform = get_transform(model)
        dataset, _ = get_dataset(dataset_name, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        scores[model] = get_score(model, dataset_name, data_loader)
        print(f"{model}: {scores[model]}")
    print(sorted(scores, key=lambda x: x[1][0]))


def get_score(model, dataset_name, data_loader):
    print(f"Evaluating {model}...")
    net = torch.load(f"models/fine_tuned_{model}_{dataset_name}.pth").to(device)
    fc_layer = get_fc_layer(model, net)
    F, Y, accuracy = forward(data_loader, net, fc_layer)

    logme = LogME(is_regression=False)
    return logme.fit(F.numpy(), Y.numpy()), accuracy


def get_transform(model):
    image_size = 299 if model == "inception_v3" else 224
    return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


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


def get_fc_layer(model, net):
    if model in pretrained_models[0:5]:
        return net.fc
    elif model in pretrained_models[5:8]:
        return net.classifier
    return net.classifier[-1]


def get_dataset(dataset_name: str, transform):
    if dataset_name == "Aircraft":
        return datasets.FGVCAircraft(root=data_path + dataset_name, download=True, transform=transform), 100
    if dataset_name == "Caltech":
        return datasets.Caltech101(root=data_path + dataset_name, download=True, transform=transform), 101
    if dataset_name == "Cars":
        return datasets.StanfordCars(root=data_path + dataset_name, download=False, transform=transform, split="test"), 196  # automatic download is not working
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=data_path + dataset_name, download=True, transform=transform, train=False), 10
    if dataset_name == "CIFAR100":
        return datasets.CIFAR100(root=data_path + dataset_name, download=True, transform=transform, train=False), 100
    if dataset_name == "DTD":
        return datasets.DTD(root=data_path + dataset_name, download=True, transform=transform, split="val"), 47
    if dataset_name == "Pets":
        return datasets.OxfordIIITPet(root=data_path + dataset_name, download=True, transform=transform), 37
    if dataset_name == "SUN":
        return datasets.SUN397(root=data_path + dataset_name, download=True, transform=transform), 397


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


def fine_tune(dataset_name:str):
    for model_name in pretrained_models:
        if os.path.exists(f"models/fine_tuned_{model_name}_{dataset_name}.pth"):
            print(f'Fine-tuned {model_name} on {dataset_name} already exists.')
            continue
        print(f"Fine-tuning {model_name} on {dataset_name}...")
        dataset, num_of_classes = get_dataset(dataset_name, get_transform(model_name))
        net = get_net(model_name)
        for param in net.parameters():
            param.requires_grad = False
        fc_layer = get_fc_layer(model_name, net)
        fc_layer.out_features = num_of_classes
        fc_layer.requires_grad_(True)
        net.to(device)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        optimizer = torch.optim.SGD(fc_layer.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        min_loss = float("inf")
        net.train()
        for epoch in range(epoches):
            train_loss = 0.
            start_time = time.time()
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if model_name == "inception_v3":
                    outputs, _ = net(inputs)
                else:
                    outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(data_loader)
            scheduler.step(train_loss)
            if train_loss < min_loss:
                min_loss = train_loss
                torch.save(net.state_dict(), f"models/fine_tuned_{model_name}_{dataset_name}.pth")
                print(f"Fine-tuned {model_name} on {dataset_name} saved on epoch {epoch+1}.")
            print(f"Epoch {epoch+1}/{epoches}, Train Loss: {train_loss:.4f},"
                  f" Remaining Time: {(epoches-epoch-1)*(time.time()-start_time)/60:.2f} minutes")
        print(f"Fine-tuning {model_name} on {dataset_name} completed.")
        print("-" * 50)


if __name__ == '__main__':
    for dataset_name in dataset_names:
        main(dataset_name)
        print("-" * 50)