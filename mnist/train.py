import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.model import NeuralNetwork
from src.dataset import MnistDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습 장치")
args = parser.parse_args()


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer) -> None:
    """ MNIst로 뉴럴넷 훈련
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련장치
    :type device: str
    :param loss_fn: 훈련에 사용되는 오차함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마지어
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss:{loss:7f} [{current:5d/{size:5d}}]')

def valid_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
    """MNISt 데이터셋으로 뉴럴넷 성능 테스트

    :param dataloader: 파이토치 데이터로더
    :type dataloader: dataloader
    :param device: 훈련에 사용되는장치
    :type device: str
    :param model: 훈련에  사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차함수
    :type loss_fn: nn.Module
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)

            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:>8f} \n')

    def train(device:str):
        num_classes = 10
        batch_size = 32
        epochs = 5
        lr = 1e-3

        """학습/추론 파이토치 파이프라인
        :param batch_size: 학습및 데이터셋 배치 크기
        :type batch_size: int
        :param epoch: 훈랸 횟수
        :type epoch: int
        """
        trainset = MnistDataset("./data/MNIST Dataset JPG format/MNIST - JPG - training")
        testset = MnistDataset("./data/MNIST Dataset JPG format/MNIST - JPG - testing")

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = NeuralNetwork(num_classes=num_classes).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        for t in range():
            print(f'Epoch {t+1}\n-----------')
            train_one_epoch(train_loader, device, model, loss_fn, optimizer)
            valid_one_epoch(test_loader, device, model, loss_fn)
        print('Done')

        torch.save(model.state_dict(), 'mnist-net.pth')
        print('Saved pytorch model state to mnist-met.pth')


    if __name__ == "__main__":
        train(device=args.device)


