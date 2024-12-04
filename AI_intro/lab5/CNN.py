import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import time

# pip install torch==2.3.0 torchvision==0.18.0 -f https://mirrors.aliyun.com/pytorch/wheels/cu121

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(3, 6, kernel_size=5),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # [64 * 224 * 224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       # [64 * 112 * 112]

            # nn.Conv2d(6, 16, kernel_size=5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [128 * 112 * 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       # [128 * 56 * 56]

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # [256 * 56 * 56]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       # [256 * 28 * 28]

            nn.Conv2d(256, 512, kernel_size=3, padding=1), # [512 * 28 * 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       # [512 * 14 * 14]

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # [512 * 14 * 14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            # nn.Linear(16 * 5 * 5, 120),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(120, 84),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(84, 10),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.layer(x)


def main():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # hyperparameters
    # 1. B = 4
    B = 128
    epoches = 100
    lr = 1e-3

    # load data
    train_dataset = CIFAR10(root="E:/Datasets/CIFAR10",
                            train=True,
                            transform=transform_train,
                            download=True)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=B,
                                   shuffle=True,
                                   num_workers=2)
    test_dataset = CIFAR10(root="E:/Datasets/CIFAR10",
                           train=False,
                           transform=transform_test,
                           download=True)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=B,
                                  shuffle=False,
                                  num_workers=2)
    net = CNN()
    print(net)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # 动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=1e-6)

    start_time = time.time()
    min_loss = 1e10
    train_losses = []
    test_losses = []
    test_accuracies = {classname: 0. for classname in classes}
    max_accuracy = 0.
    for epoch in range(epoches):
        # train
        train_loss = 0.
        train_start_time = time.time()
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_data_loader))
        print(f"Train Epoch {epoch+1}:\n\tloss: {train_loss/len(train_data_loader)}\n\ttime: {time.time()-train_start_time:.2f}s")

        # test
        test_loss = 0.
        test_start_time = time.time()
        correct_predictions = {classname: 0 for classname in classes}
        correct = 0
        total_predictions = {classname: 0 for classname in classes}
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_data_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_predictions[classes[label]] += 1
                        correct += 1
                    total_predictions[classes[label]] += 1
                    total += 1
        print(f"Test Epoch {epoch+1}:\n\tloss: {test_loss/len(test_data_loader)}\n\ttime: {time.time()-test_start_time:.2f}s")
        for classname in classes:
            test_accuracies[classname] = correct_predictions[classname]/total_predictions[classname]
            print(f"Accuracy for {classname}: {100*correct_predictions[classname]/total_predictions[classname]:.2f}%")
        print(f"Overall accuracy: {100 * correct / total:.2f}%")
        max_accuracy = max(max_accuracy, correct / total)
        print("-" * 50)
        test_losses.append(test_loss/len(test_data_loader))

        # adjust learning rate
        scheduler.step(test_loss/len(test_data_loader))
        # save model
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch,
            # 'lr': lr,
            'lr': optimizer.param_groups[0]['lr'],
        }
        if min_loss > test_loss:
            min_loss = test_loss
            torch.save(checkpoint, './models/best_model.pth')
        torch.save(checkpoint, './models/latest_model.pth')
        print(f"Model saved at epoch {epoch + 1}")
    print(f"Training finished in {time.time() - start_time:.2f}s")
    print(f'Best accuracy is {100 * max_accuracy:.2f}%')

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot loss
    x = np.arange(1, epoches + 1)
    axes[0].plot(x, train_losses, label='train')
    axes[0].plot(x, test_losses, label='test')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    # plot accuracy
    axes[1].bar(classes, [test_accuracies[classname] for classname in classes])
    axes[1].set_xlabel('class')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()