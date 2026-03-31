from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist_dataset(data_path, batch_size: int):
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        transform=transformation,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        transform=transformation,
        download=True
    )

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader, test_dataloader