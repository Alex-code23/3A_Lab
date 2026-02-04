
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


from typing import Tuple

def load_image(path: str, resize: Tuple[int, int]=(128, 128)) -> np.ndarray:
    # Load an image from path, resize to the desired resolution and return a normalized
    # numpy array of shape (H, W, 3) with values in [0,1].
    img = Image.open(path).convert('RGB')
    img = img.resize(resize)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    return img_np


def generate_synthetic_image(resolution: int = 128) -> np.ndarray:
    # Generate a synthetic RGB image using sinusoidal patterns.  This avoids external
    # dependencies and provides a known ground truth with high‑frequency content.
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xv, yv = np.meshgrid(x, y)
    img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    # Red channel: high‑frequency checkerboard pattern
    img[..., 0] = (np.sin(2 * np.pi * 6 * xv) * np.sin(2 * np.pi * 6 * yv) + 1) / 2
    # Green channel: low‑frequency radial gradient
    r = np.sqrt((xv - 0.5) ** 2 + (yv - 0.5) ** 2)
    img[..., 1] = 1 - r / np.max(r)
    # Blue channel: diagonal stripes
    img[..., 2] = (np.sin(2 * np.pi * 4 * (xv + yv)) + 1) / 2
    return img


def prepare_dataset(img: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Given an image array of shape (H, W, 3), return tensors:
    #  coords: (N, 2) containing (x,y) in [-1,1]
    #  colors: (N, 3) containing RGB in [0,1]
    h, w, c = img.shape
    ys = np.linspace(-1.0, 1.0, h)
    xs = np.linspace(-1.0, 1.0, w)
    xv, yv = np.meshgrid(xs, ys)
    coords = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
    colors = img.reshape(-1, c).astype(np.float32)
    coords = torch.from_numpy(coords).to(device)
    colors = torch.from_numpy(colors).to(device)
    return coords, colors

###########################################################
### TODO: choose your own image or generate synthetic image
###########################################################


import numpy as np
import matplotlib.pyplot as plt

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    ###########################################################
    ### TODO: Complete mse metric
    ###########################################################
    return F.mse_loss(pred, target).item()

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    ###########################################################
    ### TODO: Complete psnr metric
    ###########################################################
    mse_value = F.mse_loss(pred, target).item()
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse_value)


def try_ssim(pred_img: np.ndarray, gt_img: np.ndarray) -> float | None:
    # Optional: requires scikit-image. Returns None if unavailable.
    ###########################################################
    ### Optional TODO: Complete ssim metric
    ###########################################################
    return None

# train_model helper function
def train_model(model: nn.Module,
                coords: torch.Tensor,
                colors: torch.Tensor,
                num_iters: int = 500,
                lr: float = 1e-3,
                loss_type: str = "mse",
                log_every: int = 200) -> tuple[nn.Module, dict]:

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    history = {"step": [], "loss": [], "psnr": []}

    for it in range(1, num_iters + 1):
        ###########################################################
        ### TODO: complete the training code of Neural Field on MLP
        ###########################################################
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs, colors)
        loss.backward()
        optimizer.step()
        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                pred_colors = model(coords)
                current_loss = criterion(pred_colors, colors).item()
                current_psnr = psnr(pred_colors, colors)
                history["step"].append(it)
                history["loss"].append(current_loss)
                history["psnr"].append(current_psnr)
                print(f"Iter {it}/{num_iters}, Loss: {current_loss:.6f}, PSNR: {current_psnr:.2f} dB")


    return model, history


class SineLayer(nn.Module):
    # A single fully connected layer with sine activation and specialized initialization
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        ###########################################################
        ### TODO: complete the SineLayer class
        ###########################################################
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: weights ~ U(-1/in_features, 1/in_features)
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Subsequent layers: weights ~ U(-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0)
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-np.pi, np.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    ###########################################################
    ### TODO: define the SIREN class
    ### HINT: last layer should be linear to map to [0,1]
    ###########################################################
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 5, omega_0: float = 30.0):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        layers.append(nn.Linear(hidden_dim, out_dim))  # Last layer is linear
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
        return torch.sigmoid(self.layers[-1](x))  # sigmoid bounds outputs in [0,1]

    
if __name__ == "__main__":
    # Use GPU if available
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    image_path = None #"image.jpg"  # e.g., 'your_image.png' or None for synthetic
    resolution = 128 

    if image_path is not None:
        img_np = load_image(image_path, resize=(resolution, resolution))
    else:
        img_np = generate_synthetic_image(resolution)

    coords, colors = prepare_dataset(img_np, device)
    print(f"Dataset: {coords.shape[0]} samples, coordinate tensor shape {coords.shape}, color tensor shape {colors.shape}")

    # Train a SIREN on raw coordinates (no positional encoding needed)
    hidden_dim_siren = 256
    num_layers_siren = 5
    omega_0 = 30.0
    # Train and plot the training dynamics of the SIREN neural field model.
    model = SIREN(in_dim=2, hidden_dim=hidden_dim_siren, out_dim=3, num_layers=num_layers_siren, omega_0=omega_0).to(device)
    num_iters = 4000
    lr = 1e-4
    trained_model, history = train_model(model, coords, colors, num_iters=num_iters, lr=lr, loss_type="mse", log_every=200)


    # Show the final result of the SIREN neural field model.
    trained_model.eval()
    with torch.no_grad():
        pred_colors = trained_model(coords).cpu().numpy()
    recon_img = pred_colors.reshape(resolution, resolution, 3)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)            
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis('off')
    plt.subplot(1, 2, 2)    
    plt.title("SIREN Reconstructed Image")
    plt.imshow(recon_img)
    plt.axis('off')
    plt.show()

    # Load the model in the folder 
    folder = "CSC_52087/siren_model.pth"
    torch.save(trained_model.state_dict(), folder)