#Train a β-VAE with at least two different β values

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

#CONFIGURATION
LATENT_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
IMG_SIZE = 64
DATA_PATH = 'C:\\Users\\hegde\\OneDrive\\Desktop\\MS AI\\CV\\P3_VAE\\CelebA\\img_align_celeba' 

# We will train two separate models with these beta values
BETAS_TO_TEST = [4.0, 10.0] 

#DATASET & MODEL
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name)
            if self.transform: image = self.transform(image)
            return image
        except: return torch.zeros(3, IMG_SIZE, IMG_SIZE)

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)
        )
        self.flatten_size = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.flatten_size)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoder_input = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(decoder_input), mu, logvar
    
    def decode(self, z):
        decoder_input = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(decoder_input)

#BETA LOSS FUNCTION
def beta_loss_function(recon_x, x, mu, logvar, beta):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # The magic happens here: Multiply KLD by Beta
    return BCE + (beta * KLD)

#VISUALIZATION FUNCTION
def generate_traversals(model, device, beta_val):
    print(f"Generating Traversals for Beta={beta_val}...")
    model.eval()
    
    # We tweak 5 dimensions to see if they are disentangled
    target_dims = [0, 5, 10, 15, 20] 
    range_vals = np.linspace(-3, 3, 10)
    
    # Use a fixed random seed vector for consistency
    z_base = torch.randn(1, LATENT_DIM).to(device)
    
    plt.figure(figsize=(15, 8))
    with torch.no_grad():
        for row, dim_idx in enumerate(target_dims):
            z_traversal = []
            for val in range_vals:
                z_new = z_base.clone()
                z_new[0, dim_idx] = val
                z_traversal.append(z_new)
            
            z_stack = torch.cat(z_traversal)
            out_images = model.decode(z_stack)
            
            grid = utils.make_grid(out_images.cpu(), nrow=10, padding=2)
            plt.subplot(len(target_dims), 1, row + 1)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"Beta {beta_val}: Varying Dimension {dim_idx}")
    
    plt.tight_layout()
    plt.savefig(f"beta_{int(beta_val)}_traversals.png")
    plt.close()

#MAIN LOOP
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = CelebADataset(root_dir=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

    #LOOP OVER BETAS
    for beta in BETAS_TO_TEST:
        print(f"\n\n===================================")
        print(f"   STARTING TRAINING FOR BETA = {beta}")
        print(f"===================================")
        
        # Initialize a fresh model for each Beta
        model = VAE(latent_dim=LATENT_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            epoch_start = time.time()
            
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                
                # Use the Beta Loss Function
                loss = beta_loss_function(recon_batch, data, mu, logvar, beta)
                
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            
            avg_loss = total_loss / len(dataset)
            print(f"Beta {beta} | Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.1f} | Time: {time.time()-epoch_start:.1f}s")
            
        # Save Model
        save_path = f"vae_beta_{int(beta)}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
        
        # Generate Visualization
        generate_traversals(model, device, beta)

    print("\nDone! Check 'beta_4_traversals.png' and 'beta_10_traversals.png'.")