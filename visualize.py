# Latent Space Exploration

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
LATENT_DIM = 128
IMG_SIZE = 64
DATA_PATH = 'C:\\Users\\hegde\\OneDrive\\Desktop\\MS AI\\CV\\P3_VAE\\CelebA\\img_align_celeba'
MODEL_PATH = "vae_celeba.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RE-DEFINE CLASSES (Must match training exactly) ---
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform: image = self.transform(image)
        return image

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

    # Helper to decode a latent vector z directly
    def decode(self, z):
        decoder_input = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(decoder_input)

# --- LOAD MODEL ---
print("Loading model...")
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() # Set to evaluation mode

# --- LOAD DATA ---
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
dataset = CelebADataset(root_dir=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get a batch of images
data_iter = iter(dataloader)
images = next(data_iter).to(DEVICE)

# ==========================================
# PART 1: VISUALIZE RECONSTRUCTIONS (5 images)
# ==========================================
print("Generating Reconstructions...")
num_recon = 5
with torch.no_grad():
    originals = images[:num_recon]
    reconstructions, _, _ = model(originals)
    
    # Stitch them together: Top row = Original, Bottom = Recon
    comparison = torch.cat([originals, reconstructions])
    grid = utils.make_grid(comparison.cpu(), nrow=num_recon, padding=2)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Top: Original | Bottom: Reconstructed")
    plt.savefig("task2_reconstructions.png")
    plt.show()

# ==========================================
# PART 2: LATENT INTERPOLATION (Morphing)
# ==========================================
print("Generating Interpolations...")
# We will interpolate between pairs: image[0]->image[1], image[2]->image[3], etc.
num_pairs = 3
steps = 8 # How many frames in the morph

plt.figure(figsize=(15, 6))

with torch.no_grad():
    for i in range(num_pairs):
        # 1. Get Latent Vectors for Start (A) and End (B)
        imgA = images[i*2].unsqueeze(0)
        imgB = images[i*2+1].unsqueeze(0)
        
        # Encode to get mu (we use mu as the "clean" representation)
        _, muA, _ = model(imgA)
        _, muB, _ = model(imgB)
        
        # 2. Interpolate
        # Linear Interpolation formula: z = (1-alpha)*A + alpha*B
        z_steps = []
        for alpha in np.linspace(0, 1, steps):
            z_step = (1 - alpha) * muA + alpha * muB
            z_steps.append(z_step)
        
        z_stack = torch.cat(z_steps) # Shape: [steps, 128]
        
        # 3. Decode the path
        morph_images = model.decode(z_stack)
        
        # 4. Plotting
        grid = utils.make_grid(morph_images.cpu(), nrow=steps, padding=2)
        plt.subplot(num_pairs, 1, i + 1)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Interpolation Pair {i+1}")

plt.tight_layout()
plt.savefig("task2_interpolations.png")
plt.show()

# ==========================================
# PART 3: LATENT TRAVERSAL (Disentanglement)
# ==========================================
print("Generating Traversals...")
# We take ONE face and tweak specific dimensions of its latent code
target_img = images[0].unsqueeze(0)

# Encode to get the base latent vector
with torch.no_grad():
    _, z_base, _ = model(target_img)

# We will try to vary 5 random dimensions to see if they correspond to features
# Note: In standard VAEs, dimensions aren't perfectly disentangled, but you might see changes.
target_dims = [0, 10, 25, 50, 100] # Arbitrary indices
range_vals = np.linspace(-3, 3, 10) # From -3 sigma to +3 sigma

plt.figure(figsize=(15, 8))

with torch.no_grad():
    for row, dim_idx in enumerate(target_dims):
        z_traversal = []
        
        for val in range_vals:
            z_new = z_base.clone()
            z_new[0, dim_idx] = val # Override just this one dimension
            z_traversal.append(z_new)
            
        z_stack = torch.cat(z_traversal)
        out_images = model.decode(z_stack)
        
        grid = utils.make_grid(out_images.cpu(), nrow=10, padding=2)
        plt.subplot(len(target_dims), 1, row + 1)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Varying Latent Dimension {dim_idx}")

plt.tight_layout()
plt.savefig("task2_traversals.png")
plt.show()

print("Done! All images saved.")