import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import time


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            # 1. nn.Conv2d(3, 6, kernel_size=5),
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 1. nn.Conv2d(6, 16, kernel_size=5),
            nn.Conv2d(32, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            # 1. nn.Linear(16 * 5 * 5, 120),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 1. nn.Linear(120, 84),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 1.nn.Linear(84, 10),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.layer(x)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # hyperparameters
    # 1. B = 4
    B = 32
    epoches = 100
    lr = 1e-3

    # load data
    train_dataset = CIFAR10(root="E:/Datasets/CIFAR10",
                            train=True,
                            transform=transform,
                            download=True)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=B,
                                   shuffle=True,
                                   num_workers=2)
    test_dataset = CIFAR10(root="E:/Datasets/CIFAR10",
                           train=False,
                           transform=transform,
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
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    min_loss = 1e10
    train_losses = []
    test_losses = []
    test_accuracies = {classname: 0. for classname in classes}
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
        total_predictions = {classname: 0 for classname in classes}
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
                    total_predictions[classes[label]] += 1
        print(f"Test Epoch {epoch+1}:\n\tloss: {test_loss/len(test_data_loader)}\n\ttime: {time.time()-test_start_time:.2f}s")
        for classname in classes:
            test_accuracies[classname] = correct_predictions[classname]/total_predictions[classname]
            print(f"Accuracy for {classname}: {100*correct_predictions[classname]/total_predictions[classname]:.2f}%")
        test_losses.append(test_loss/len(test_data_loader))

        # save model
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch,
            'lr': lr,
        }
        if min_loss > test_loss:
            min_loss = test_loss
            torch.save(checkpoint, './models/best_model.pth')
        torch.save(checkpoint, './models/latest_model.pth')
        print(f"Model saved at epoch {epoch+1}")
    print(f"Training finished in {time.time()-start_time:.2f}s")

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