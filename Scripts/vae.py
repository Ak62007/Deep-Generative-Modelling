import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

current_dir = Path.cwd()
project_root = current_dir.parent

transformation = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
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
    def __init__(self, latent_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(784, 400)
        self.layer2 = nn.Linear(400, 100)
        self.mu_head = nn.Linear(100, latent_dim)
        self.log_var_head = nn.Linear(100, latent_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        mu = self.mu_head(x)
        log_var = self.log_var_head(x)
        return mu, log_var
    
def reparametarization(mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(mu)
    return mu + sigma * epsilon

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.layer1 = nn.Linear(latent_dim, 100)
        self.layer2 = nn.Linear(100, 400)
        self.layer3 = nn.Linear(400, 784)
        
    def forward(self, z):
        z = self.layer1(z)
        z = F.relu(z)
        z = self.layer2(z)
        z = F.relu(z)
        z = self.layer3(z)
        x = torch.sigmoid(z)
        return x
    
# VAE Network
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparametarization(mu=mu, log_var=log_var)
        x_constructed = self.decoder(z)
        return x_constructed, mu, log_var
    
# Loss Function
def vae_loss(x_constructed, x_original, mu, log_var):
    # reconstruction error
    recons_err = F.binary_cross_entropy(x_constructed, x_original, reduction='sum')
    # Kl divergence regularizer
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))
    loss = (recons_err + kl_loss)/x_original.size(0)
    return loss


if __name__ == "__main__":
    
    latent_dim = 20
    num_epochs = 50
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for image, _ in train_dataloader:
            x_contructed, mu, log_var = model.forward(image)
            loss = vae_loss(x_contructed, image, mu, log_var)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss/len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss.item():.2f}")