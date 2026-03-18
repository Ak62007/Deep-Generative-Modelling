import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

cwd = Path.cwd()
project_root = cwd.parent

transformation = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(
    root=project_root / "data",
    train=True,
    transform=transformation,
    download=True
)

test_dataset = datasets.MNIST(
    root=project_root / "data",
    train=False,
    transform=transformation,
    download=True
)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, D):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4*D,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=4*D,
            out_channels=2*D,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_layer3 = nn.Conv2d(
            in_channels=2*D,
            out_channels=D,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()
        
    def forward(self, X):
        X = self.conv_layer1(X)
        X = self.relu(X)
        X = self.conv_layer2(X)
        X = self.relu(X)
        X = self.conv_layer3(X)
        return X
    
# CodeBook
class CodeBook(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1/K, 1/K)

    def forward(self, X:torch.tensor):
        B, D_c, H, W = X.shape

        X_permuted = X.permute(0, 2, 3, 1).contiguous()
        X_flattened = X_permuted.view(-1, D_c)

        distances = (X_flattened**2).sum(dim=1, keepdim=True) + ((self.embedding.weight**2).sum(dim=1, keepdim=True)).T - 2*(X_flattened @ self.embedding.weight.T)
        min_dis_ind = distances.argmin(dim=1)

        zq_flattened = self.embedding.weight[min_dis_ind]
        zq_straight_flattened = X_flattened + (zq_flattened - X_flattened).detach()

        zq_straight = zq_straight_flattened.view(B, H, W, D_c).permute(0, 3, 1, 2).contiguous()
        zq = zq_flattened.view(B, H, W, D_c).permute(0, 3, 1, 2).contiguous()

        return zq_straight, zq
    

# Decoder
class Decoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.inv_conv_layer1 = nn.ConvTranspose2d(
            in_channels=D,
            out_channels=2*D,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.inv_conv_layer2 = nn.ConvTranspose2d(
            in_channels=2*D,
            out_channels=4*D,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.inv_conv_layer3 = nn.ConvTranspose2d(
            in_channels=4*D,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        
    def forward(self, X):
        X = self.inv_conv_layer1(X)
        X = self.relu(X)
        X = self.inv_conv_layer2(X)
        X = self.relu(X)
        X = self.inv_conv_layer3(X)
        X = self.sigmoid(X)
        return X
    

# VQ_VAE
class VQ_VAE(nn.Module):
    def __init__(self, in_channels, D, K):
        super().__init__()
        self.encoder = Encoder(in_channels, D)
        self.codebook = CodeBook(K, D)
        self.decoder = Decoder(D)
        
    def forward(self, X):
        ze = self.encoder(X)
        zq_straight, zq_raw = self.codebook(ze)
        generated_img = self.decoder(zq_straight)
        return generated_img, ze, zq_raw
    
# Cost Function
def cost_fn(
    original_img,
    generated_img,
    ze,
    zq,
    beta: float = 0.25
):
    # reconstruction error
    term1 = F.binary_cross_entropy(generated_img, original_img)
    
    # codebook loss
    term2 = ((ze.detach() - zq)**2).mean()
    
    # commitment loss
    term3 = ((ze - zq.detach())**2).mean()
    
    # print(f"term 1: {term1}")
    # print(f"term 2: {term2}")
    # print(f"term 3: {term3}")
    
    loss = term1 + term2 + beta*term3
    
    return loss

def train(model:VQ_VAE, dataloader: DataLoader, optimizer, epochs: int):
    model.train()
    for i in range(epochs):
        total_loss = 0
        for batch in dataloader:
            img, _  = batch
            img = img.to(device)
            
            # forward pass
            generated_img, ze, zq = model.forward(img)
            # calculating loss
            loss = cost_fn(
                original_img=img,
                generated_img=generated_img,
                ze=ze,
                zq=zq
            )
            # total loss
            total_loss += loss
            # zero the gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {i+1}: Loss: {avg_loss.item():.2f}")
        
    return model


def visualize_reconstructions(model, dataloader, n=10):
    # Switching to eval mode
    model.eval()
    
    with torch.no_grad():
        figs, axes = plt.subplots(nrows=2, ncols=n, figsize=(12,3))
        # getting one batch
        original_imgs, _ = next(iter(dataloader))
        # inference
        generated_imgs, _, _ = model.forward(original_imgs)
        
        for i in range(n):
            axes[0, i].imshow(original_imgs[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(generated_imgs[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            
        plt.show()
        
def visualize_codebook(model:VQ_VAE, K, D, n=64):
    if n > K:
        raise ValueError(f"n should be less than or equal {K}")
    model.eval()
    
    with torch.no_grad():
        figs, axes = plt.subplots(nrows=8, ncols=8, figsize=(8,8))
        
        for i in range(n):
            feature = model.codebook.embedding.weight[i]
            ones_vec = torch.ones(size=[D, 7, 7])
            feature = feature.unsqueeze(dim=1)
            feature = feature.unsqueeze(dim=1)
            dec_input = feature * ones_vec
            generated_feature = model.decoder.forward(dec_input)
            generated_feature = generated_feature.squeeze()
            
            row = i // 8
            col = i % 8
            axes[row, col].imshow(generated_feature, cmap='gray')
            axes[row, col].axis('off')
        
        plt.show()
        
def generate_images(model:VQ_VAE, K:int, n:int=25):
    model.eval()
    
    with torch.no_grad():
        figs, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8))
        for i in range(n):
            # randomly sampling some features from the codebook
            random_idxs = torch.randint(low=0, high=K, size=[7, 7])
            
            # getting the features
            random_features = model.codebook.embedding.weight[random_idxs]
            
            # reshaping
            random_features = random_features.permute(2, 0, 1)
            random_features = random_features.unsqueeze(0)
            
            # generating the image
            generated_img = model.decoder.forward(random_features)
            
            row = i // 5
            col = i % 5
            axes[row, col].imshow(generated_img.squeeze(), cmap='gray')
            axes[row, col].axis('off')
            
        plt.show()
        
# PIXELCNN
def encode_dataset(model:VQ_VAE, dataloader:DataLoader):
    model.eval()
    all_indices = []

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device) 
            ze = model.encoder.forward(imgs)
            B, D_c, H, W = ze.shape

            ze_flattened = ze.permute(0, 2, 3, 1).contiguous().view(-1, D_c)

            distances = (ze_flattened**2).sum(dim=1, keepdim=True) + ((model.codebook.embedding.weight**2).sum(dim=1, keepdim=True)).T - 2*(ze_flattened @ model.codebook.embedding.weight.T)
            min_dis_ind = distances.argmin(dim=1)
            min_dis_ind = min_dis_ind.view(B, H, W)

            all_indices.append(min_dis_ind.cpu())

    return torch.cat(all_indices, dim=0)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mask_type = mask_type
        self.register_buffer('mask', torch.ones_like(self.weight))
        self._create_mask()

    def _create_mask(self):
        k_h, k_w = self.kernel_size
        self.mask[:, :, k_h//2+1:, :] = 0

        if self.mask_type == 'A':
            self.mask[:, :, k_h//2, k_w//2:] = 0
        else:
            self.mask[:, :, k_h//2, k_w//2+1:] = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    
class MaskedResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = MaskedConv2d('B', in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=1)
        self.conv2 = MaskedConv2d('B', in_channels=hidden_dim//2, out_channels=hidden_dim//2, kernel_size=3, padding=1)
        self.conv3 = MaskedConv2d('B', in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return self.relu(out + residual)
    
    
class PixelCNN(nn.Module):
    def __init__(self, K:int, embed_dim:int, hidden_dim:int, num_res_blocks:int=5):
        super().__init__()
        self.relu = nn.ReLU()
        self.embed_layer = nn.Embedding(K, embed_dim)

        # Initial Type A convolution
        self.initial_conv = MaskedConv2d('A', in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # Stack of Residual Blocks
        self.res_blocks = nn.Sequential(
            *[MaskedResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Final output layers (Type B)
        self.final_conv1 = MaskedConv2d('B', in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.final_conv2 = MaskedConv2d('B', in_channels=hidden_dim, out_channels=K, kernel_size=1)

    def forward(self, X:torch.tensor):
        X = self.embed_layer(X)
        X = X.permute(0, 3, 1, 2).contiguous()

        X = self.relu(self.initial_conv(X))
        X = self.res_blocks(X)

        X = self.relu(self.final_conv1(X))
        X = self.final_conv2(X)
        return X
    
def train_pixelcnn(pixelcnn: nn.Module, indices: torch.tensor, optimizer, scheduler, epochs):
    pixelcnn.train()
    dataset = TensorDataset(indices)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            indices_batch, = batch
            indices_batch = indices_batch.to(device)

            output = pixelcnn.forward(indices_batch)
            loss = criterion(output, indices_batch.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # Get the current learning rate for logging
        current_lr = scheduler.get_last_lr()[0]

        print(f"PixelCNN Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(dataloader):.4f} | LR: {current_lr:.6f}")

    return pixelcnn

def generate_with_pixelcnn(pixelcnn:nn.Module, vqvae:VQ_VAE, n=25, temperature=0.8): 
    pixelcnn.eval()
    vqvae.eval()

    with torch.no_grad():
        grid = torch.zeros(n, 7, 7).long().to(device)

        for i in range(7):
            for j in range(7):
                distribution = pixelcnn.forward(grid)

                logits = distribution[:, :, i, j] / temperature

                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                grid[:, i, j] = sampled.squeeze()

        sampled_features = vqvae.codebook.embedding.weight[grid]
        sampled_features = sampled_features.permute(0, 3, 1, 2).contiguous()
        generated_images = vqvae.decoder.forward(sampled_features)

        generated_images = generated_images.cpu()

        figs, axes = plt.subplots(nrows=5, ncols=5, figsize=(8,8))
        for i in range(n):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(generated_images[i].squeeze(), cmap='gray')
            axes[row, col].axis('off')

        plt.show()