#Baseline VAE Implementation for CelebA Dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

#OPTIMIZATION
# This enables the auto-tuner to find the best algorithm for your hardware
torch.backends.cudnn.benchmark = True

#Hyperparameters
LATENT_DIM = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50 
IMG_SIZE = 64

# PATH SETUP
DATA_PATH = 'C:\\Users\\hegde\\OneDrive\\Desktop\\MS AI\\CV\\P3_VAE\\CelebA\\img_align_celeba' 

#CUSTOM DATASET
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Path not found: {os.path.abspath(root_dir)}")
            
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {os.path.abspath(root_dir)}.")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # Return a dummy tensor so training doesn't crash on one bad file
            return torch.zeros(3, IMG_SIZE, IMG_SIZE) 

#VAE MODEL ARCHITECTURE
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.flatten_size = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.flatten_size)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 256, 4, 4)
        reconstruction = self.decoder(decoder_input)
        return reconstruction, mu, logvar

#LOSS FUNCTION
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#MAIN EXECUTION BLOCK
if __name__ == '__main__':
    # --- GPU CHECK ---
    print("\n--- HARDWARE CHECK ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ SUCCESS: GPU Found: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDNN Benchmark: Enabled")
    else:
        device = torch.device("cpu")
        print("❌ WARNING: GPU NOT Found. Training will be VERY slow.")
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Initialize Dataset
    print(f"\nLoading data from: {os.path.abspath(DATA_PATH)}")
    try:
        dataset = CelebADataset(root_dir=DATA_PATH, transform=transform)
        
        # OPTIMIZATION: pin_memory=True speeds up transfer from CPU to GPU
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Data loaded. Found {len(dataset)} images.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

    # Model Setup
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    train_losses = []
    print("\nStarting Training...")
    
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} running...", end=" ")
        
        for batch_idx, data in enumerate(dataloader):
            # Move data to GPU
            data = data.to(device, non_blocking=True) # non_blocking helps speed
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        epoch_end = time.time()
        avg_loss = total_loss / len(dataset)
        train_losses.append(avg_loss)
        
        print(f"\n   -> Time: {epoch_end - epoch_start:.1f}s | Average loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes.")

    # Save Model
    torch.save(model.state_dict(), "vae_celeba.pth")
    print("Model saved to vae_celeba.pth")

    #VISUALIZATION
    print("Generating visualizations...")
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Total Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    # 2. Reconstructions
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        images = next(data_iter).to(device)
        
        recon_images, _, _ = model(images)
        
        images = images.cpu()
        recon_images = recon_images.cpu()
        
        n_view = 8
        comparison = torch.cat([images[:n_view], recon_images[:n_view]])
        grid = utils.make_grid(comparison, nrow=n_view)
        
        plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title("Original (Top) vs Reconstructed (Bottom)")
        plt.savefig('reconstruction_results.png') 
        plt.show()

    print("Done!")