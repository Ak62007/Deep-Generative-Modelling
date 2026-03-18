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

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

g_lr = 2e-4
d_lr = 1e-4
batch_size = 128
noise_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, noise_size: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features=noise_size, out_features=512*7*7)
        self.layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(256)
        self.batch_norm_2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, X):
        X = self.layer1(X) # (512 * 7 * 7)
        X = X.view(-1, 512, 7, 7) # (128, 512, 7, 7)
        X = self.layer2(X) # (128, 256, 14, 14)
        X = self.batch_norm_1(X)
        X = self.relu(X)
        X = self.layer3(X) # (128, 128, 28, 28)
        X = self.batch_norm_2(X)
        X = self.relu(X)
        X = self.layer4(X) # (128, 1, 28, 28)
        X = self.tanh(X)
        
        return X
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Linear(in_features=256*7*7, out_features=1024)
        self.layer4 = nn.Linear(in_features=1024, out_features=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.l_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        
    def forward(self, X):
        X = self.layer1(X)
        X = self.l_relu(X)
        X = self.layer2(X)
        X = self.batch_norm(X)
        X = self.l_relu(X)
        X = self.flatten(X)
        X = self.layer3(X)
        X = self.l_relu(X)
        X = self.layer4(X)
        X = self.sigmoid(X)
        
        return X
    
def train(
    dataloader: DataLoader,
    generator: Generator,
    discriminator: Discriminator,
    criterion,
    noise_size:int,
    device,
    g_optimizer,
    d_optimizer,
    epochs : int = 10,
    k: int = 1
):
    generator.train()
    discriminator.train()

    d_total_loss_track = []
    g_total_loss_track = []

    for epoch in range(epochs):
        d_total_loss = 0
        g_total_loss = 0

        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch, _ in progress_bar:
            current_batch_size = batch.shape[0]
            batch = batch.to(device, non_blocking=True) 

            # Train Discriminator
            for i in range(k):
                d_optimizer.zero_grad()

                # Generate fake images
                noise_sample = torch.randn(current_batch_size, noise_size, device=device)
                generated_samples = generator(noise_sample).detach() # Use () instead of .forward()

                # Create labels DIRECTLY on the device
                g_labels = torch.zeros(current_batch_size, device=device)
                r_labels = torch.ones(current_batch_size, device=device) * 0.9

                # Predictions (squeeze only the channel dimension)
                g_generator_output = discriminator(generated_samples).squeeze(1)
                r_generator_output = discriminator(batch).squeeze(1)

                # Loss
                g_cost = criterion(g_generator_output, g_labels)
                r_cost = criterion(r_generator_output, r_labels)

                d_loss = g_cost + r_cost
                d_total_loss += d_loss.item()
                d_loss.backward()
                d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            new_noise_sample = torch.randn(current_batch_size, noise_size, device=device)
            new_fake_samples = generator(new_noise_sample)
            new_d_output = discriminator(new_fake_samples).squeeze(1)

            # Generator wants to fool discriminator into outputting 1s
            new_labels = torch.ones(current_batch_size, device=device)

            new_g_cost = criterion(new_d_output, new_labels)
            g_total_loss += new_g_cost.item()
            new_g_cost.backward()
            g_optimizer.step()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{new_g_cost.item():.4f}"
            })
        g_total_loss_track.append(g_total_loss/len(dataloader))
        d_total_loss_track.append(d_total_loss/len(dataloader))
        print(f"Epoch {epoch+1}/{epochs} Completed | Avg D Loss: {d_total_loss/len(dataloader):.4f} | Avg G Loss: {g_total_loss/len(dataloader):.4f}")

    return generator, discriminator, d_total_loss_track, g_total_loss_track

def visualize(generator: Generator, noise_size, device, n=16):
    generator.eval()
    with torch.no_grad():
        noise_data = torch.randn(size=[n, noise_size]).to(device)
        output = generator.forward(noise_data)
        output = output.cpu()
        output = output.squeeze(1)
        output = (output + 1)/2

        figs, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
        for i in range(n):
            row = i // 4
            col = i % 4
            axes[row][col].imshow(output[i], cmap='gray')
            axes[row][col].axis('off')

    plt.show()
    
    
if __name__ == "__main__":
    
    generator = Generator(noise_size=noise_size).to(device)
    discriminator = Discriminator().to(device)
    criterian = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)
    
    trained_generator, trained_discriminator, track_d, track_g = train(
        dataloader=train_dataloader,
        generator=generator,
        discriminator=discriminator,
        criterion=criterian,
        noise_size=noise_size,
        device=device,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        epochs=50,
    )
    
    visualize(
        generator=trained_generator,
        noise_size=noise_size,
        device=device
    )