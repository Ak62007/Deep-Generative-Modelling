import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

cwd = Path.cwd()
project_root = cwd.parent

batch_size = 128
noise_size = 128

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = datasets.MNIST(
    root=project_root / "data",
    train=True,
    transform=transformation,
)

test_dataset = datasets.MNIST(
    root=project_root / "data",
    train=False,
    transform=transformation,
)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

# Generator
class Generator(nn.Module):
    def __init__(self, noise_size:int):
        super().__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(in_features=noise_size, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=512)
        self.layer3 = nn.Linear(in_features=512, out_features=28*28)
    
    def forward(self, X):
        X = self.layer1(X)
        X = self.relu(X)
        X = self.layer2(X)
        X = self.relu(X)
        X = self.layer3(X)
        X = self.tanh(X)
        return X
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(in_features=28*28, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, X):
        X = self.layer1(X)
        X = self.lrelu(X)
        X = self.layer2(X)
        X = self.lrelu(X)
        X = self.layer3(X)
        X = self.sigmoid(X)
        return X
    
    
def train(
    dataloader: DataLoader,
    generator: Generator,
    discriminator: Discriminator,
    criterion: nn.BCELoss,
    noise_size: int,
    device,
    g_optimizer,
    d_optimizer,
    epochs: int = 10,
    k: int = 1
    ):
    """
    Args:
        generator (Generator): generator part of the GAN
        discriminator (Discriminator): discriminator part of the GAN
        criterion (nn.BCELoss): criterion
        epochs (int, optional): no of epochs to run. Defaults to 10.
        k (int, optional): no of times discriminator should be trained. Defaults to 1.
    """
    for epoch in range(epochs):
        g_total_loss = 0
        d_total_loss = 0
        for batch, _ in dataloader:
            for _ in range(k):
                # batch of generated samples.
                batch = batch.to(device)
                batch = batch.view(batch.shape[0], -1)
                noise_sample = torch.randn(size=[batch.shape[0], noise_size]).to(device)
                generated_samples = generator.forward(noise_sample).detach()
                g_labels = torch.zeros(size=[batch.shape[0]]).squeeze().to(device)
                r_labels = torch.ones(size=[batch.shape[0]]).squeeze().to(device)
                g_generator_output = discriminator.forward(generated_samples).squeeze()
                r_generator_output = discriminator.forward(batch).squeeze()
                g_cost = criterion(g_generator_output, g_labels)
                r_cost = criterion(r_generator_output, r_labels)
                d_optimizer.zero_grad()
                d_loss = g_cost + r_cost
                d_total_loss += d_loss.item()
                d_loss.backward()
                d_optimizer.step()
            new_noise_sample = torch.randn(size=[batch.shape[0], noise_size]).to(device)
            new_fake_samples = generator.forward(new_noise_sample)
            new_d_output = discriminator.forward(new_fake_samples).squeeze()
            new_labels = torch.ones(size=[batch.shape[0]]).to(device)
            g_optimizer.zero_grad()
            new_g_cost = criterion(new_d_output, new_labels)
            g_total_loss += new_g_cost.item()
            new_g_cost.backward()
            g_optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_total_loss:.4f} | G Loss: {g_total_loss:.4f}")
    return generator, discriminator

def visualize(generator: Generator, noise_size, device, n=16):
    generator.eval()
    with torch.no_grad():
        noise_data = torch.randn(size=[n, noise_size]).to(device)
        output = generator.forward(noise_data)
        output = output.cpu()
        output = output.view(n, 28, 28)
        output = (output + 1)/2
        
        figs, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
        for i in range(n):
            row = i // 4
            col = i % 4
            axes[row][col].imshow(output[i], cmap='gray')
            axes[row][col].axis('off')
    
    plt.show()
    
    
if __name__ == "__main__":
    epochs = 3 # starting small to test if the loss is decresing or not.
    batch_size = 128
    noise_size = 128
    lr = 2e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(noise_size).to(device)
    discriminator = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    
    trained_generator, trained_discriminator = train(
        dataloader=train_dataloader,
        generator=generator,
        discriminator=discriminator,
        criterion=criterion,
        noise_size=noise_size,
        batch_size=batch_size,
        device=device,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        epochs=50,
    )
    
    