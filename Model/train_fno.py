import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm

# 1. Dataset and DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
class ChannelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (B, 4, H, W)
        B = x.shape[0]
        # Sum of squares of all channels: total probability per sample
        prob = (x ** 2).sum(dim=[1, 2, 3], keepdim=True)  # (B,1,1,1)
        norm_factor = torch.sqrt(prob + self.eps)
        return x / norm_factor
def test_and_animate(model, dataset, epoch, steps=15, save_path="animations"):
    """
    Roll out the model autoregressively for `steps` frames, using
    each prediction as the input to the next, and compare against
    ground truth densities for both particles. Also calculate % error.
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # Load complete wavefunction series for the first sample
    sample_key, _ = dataset.index[0]
    wf_np = dataset.h5[sample_key]['wavefunction'][:]  
    # shape: (frames, 2 particles, 2 channels, N, N)
    frames_available = wf_np.shape[0]

    # Prepare the very first input W(0)
    initial = wf_np[0].reshape(4, wf_np.shape[-2], wf_np.shape[-1])
    current = torch.from_numpy(initial).unsqueeze(0).to(device).float()  # (1,4,N,N)

    preds = []
    with torch.no_grad():
        for _ in range(min(steps, frames_available - 1)):
            out = model(current)          # (1,4,N,N)
            preds.append(out.squeeze(0).cpu())
            current = out

    # Build true density sequence for comparison: D(1)...D(steps)
    truths = []
    for t in range(1, 1 + len(preds)):
        real = wf_np[t][:, 0, :, :]  # (2,N,N)
        imag = wf_np[t][:, 1, :, :]
        truths.append(torch.from_numpy(real**2 + imag**2))

    # Compute predicted densities for each rollout step
    pred_dens = []
    for p in preds:
        densA = p[0]**2 + p[1]**2
        densB = p[2]**2 + p[3]**2
        pred_dens.append(torch.stack([densA, densB]))

    # Calculate % error for each particle over time
    percent_errors = []
    for pred, true in zip(pred_dens, truths):
        error_A = torch.abs(pred[0] - true[0]) / (true[0] + 1e-8)
        error_B = torch.abs(pred[1] - true[1]) / (true[1] + 1e-8)
        percent_error = 0.5 * (error_A.mean().item() + error_B.mean().item()) * 100
        percent_errors.append(percent_error)

    avg_percent_error = sum(percent_errors) / len(percent_errors)
    print(f"[test_and_animate] Average percent error: {avg_percent_error:.2f}%")

    # Set up figure: 2 rows (particles A/B), 2 cols (pred vs true)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    im_objs = []
    vmin, vmax = 0, max(truths[0].max(), pred_dens[0].max()).item()

    for i, particle in enumerate(["A", "B"]):
        im_pred = axes[i,0].imshow(pred_dens[0][i], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[i,0].set_title(f"Predicted |ψ_{particle}|²")
        im_true = axes[i,1].imshow(truths[0][i],    cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[i,1].set_title(f"True      |ψ_{particle}|²")
        im_objs.append((im_pred, im_true))

    def update(frame):
        for i in range(2):
            im_objs[i][0].set_data(pred_dens[frame][i])
            im_objs[i][1].set_data(truths[frame][i])
        fig.suptitle(f"Autoregressive step {frame+1}/{len(preds)}\nError: {percent_errors[frame]:.2f}%")
        return [obj for pair in im_objs for obj in pair]

    ani = animation.FuncAnimation(fig, update, frames=len(preds), interval=300, blit=True)
    gif_file = os.path.join(save_path, f"epoch_{epoch:03d}_autoregressive.gif")
    ani.save(gif_file, writer='pillow')
    plt.close(fig)
    print(f"[test_and_animate] Saved autoregressive GIF: {gif_file}")



class WavefunctionDataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.index = []
        for sample in self.h5.keys():
            frames = self.h5[sample]['wavefunction'].shape[0]
            for t in range(frames - 1):
                self.index.append((sample, t))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        sample, t = self.index[idx]
        wf = self.h5[sample]['wavefunction']
        # load complex wavefunction: shape (frames, 2 particles, 2 channels, N, N)
        inp = wf[t]       # next step prediction input
        tgt = wf[t + 1]   # ground truth
        # merge particle & complex channels → 4 channels
        inp = inp.reshape((4, inp.shape[-2], inp.shape[-1]))
        tgt = tgt.reshape((4, tgt.shape[-2], tgt.shape[-1]))
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = WavefunctionDataset("quantum_wavefunctions_complex.h5")
loader = DataLoader(dataset, batch_size=100, shuffle=True)

# 2. FNO Model

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, modes, modes, 2) * (1/(in_channels*out_channels)))
    def compl_mul2d(self, input_ft, weights):
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy, ioxy->boxy", input_ft, cweights)
    def forward(self, x):
        # x: (batch, C, H, W)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        out_ft = self.compl_mul2d(x_ft, self.weights)
        out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return out

# U‑Net + FNO bottleneck
class UNetFNO(nn.Module):
    def __init__(self, modes=20, width=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 64→32

        self.enc2 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, padding=1), nn.GELU(),
            nn.Conv2d(width*2, width*2, 3, padding=1), nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(2)  # 32→16

        # FNO bottleneck
        self.fno1 = SpectralConv2d(width*2, width*2, modes)
        self.fno2 = SpectralConv2d(width*2, width*2, modes)

        # Decoder
        self.up1 = nn.ConvTranspose2d(width*2, width, kernel_size=2, stride=2)  # 16→32
        self.dec1 = nn.Sequential(
            nn.Conv2d(width*2, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )

        self.up2 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)  # 32→64
        self.dec2 = nn.Sequential(
            nn.Conv2d(width*2, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )

        # Output projection
        self.outc = nn.Conv2d(width, 4, 1)
        self.out_act = ChannelNorm()
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # (b, width, 64,64)
        p1 = self.pool1(e1)     # (b, width, 32,32)

        e2 = self.enc2(p1)      # (b, width*2,32,32)
        p2 = self.pool2(e2)     # (b, width*2,16,16)

        # FNO bottleneck
        f = self.fno1(p2)
        f = F.gelu(f)
        f = self.fno2(f)
        f = F.gelu(f)           # (b, width*2,16,16)

        # Decoder
        u1 = self.up1(f)        # (b, width,32,32)
        c1 = torch.cat([u1, e2[:, :u1.size(1),:,:]], dim=1)  # skip from enc2
        d1 = self.dec1(c1)      # (b, width,32,32)

        u2 = self.up2(d1)       # (b, width,64,64)
        c2 = torch.cat([u2, e1], dim=1)  # skip from enc1
        d2 = self.dec2(c2)      # (b, width,64,64)

        out = self.outc(d2)     # (b,4,64,64)
        return self.out_act(out)
class VanillaANN(nn.Module):
    def __init__(self, grid_size, hidden_dim=4000):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = 4 * grid_size * grid_size
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(self.input_dim, self.input_dim),
            # nn.Tanh(),
            # nn.Linear(self.input_dim, self.input_dim),
            # nn.Tanh(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim),
            # nn.Linear(self.input_dim, self.input_dim),
            # nn.Tanh(),
            # nn.Linear(self.input_dim, self.input_dim),
        )
        self.out_act = ChannelNorm()
    def forward(self, x):
        # x: (batch, 4, N, N)
        b = x.size(0)
        out = self.net(x)  # (b, 4*N*N)
        return out.view(b, 4, self.grid_size, self.grid_size)

# 2. Instantiate vanilla ANN
N = 64  # Must match your data grid resolution

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, modes, modes, 2) * self.scale)

    def compl_mul2d(self, input_ft, weights):
        weights_c = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights_c)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        mx = min(self.modes, H)
        my = min(self.modes, W // 2 + 1)

        out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

class CNN_FNO_Skip_LargeKernel(nn.Module):
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, width, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(width, width*2, kernel_size=5, stride=2, padding=2),  # 64→32
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(width*2, width*4, kernel_size=5, stride=2, padding=2),  # 32→16
            nn.GELU()
        )

        self.fno1 = SpectralConv2d(width*4, width*4, modes)
        self.fno2 = SpectralConv2d(width*4, width*4, modes)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(width*4, width*2, kernel_size=5, stride=2, padding=2, output_padding=1),  # 16→32
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(width*4, width, kernel_size=5, stride=2, padding=2, output_padding=1),  # 32→64
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(width*2, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.enc1(x)  # (B, width, 64, 64)
        x2 = self.enc2(x1)  # (B, width, 64, 64)
        d1 = self.down1(x2)  # (B, width*2, 32, 32)
        d2 = self.down2(d1)  # (B, width*4, 16, 16)

        f = F.gelu(self.fno1(d2))
        f = F.gelu(self.fno2(f))

        u1 = self.up1(f)  # (B, width*2, 32, 32)
        u1_cat = torch.cat([u1, d1], dim=1)  # skip from d1

        u2 = self.up2(u1_cat)  # (B, width, 64, 64)
        u2_cat = torch.cat([u2, x2], dim=1)  # skip from x2

        return self.out_conv(u2_cat)
def probability_loss(output):
    """
    Compute the loss enforcing total probability of 1.
    Assumes output shape is (B, 4, H, W) with [ReA, ImA, ReB, ImB]
    """
    B, _, H, W = output.shape

    # Extract real/imag parts
    reA, imA, reB, imB = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

    # Compute densities
    prob_A = (reA**2 + imA**2).sum(dim=[1, 2])  # sum over H,W
    prob_B = (reB**2 + imB**2).sum(dim=[1, 2])

    # Total probability per sample (should be 1.0 for each)
    total_prob = prob_A + prob_B

    # Loss = squared deviation from 1
    loss = ((total_prob - 1.0) ** 2).mean()
    return loss

model = VanillaANN(grid_size=N, hidden_dim=2048).to(device)
# model = CNN_FNO_Skip_LargeKernel().to(device)
# model = UNetFNO().to(device)

# 3. Training Loop

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 500
for ep in range(epochs):
    model.train()
    total_loss = 0.0
    for inp, tgt in tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()* inp.size(0)
    avg_loss = total_loss / len(dataset)
    if ep%10 == 0:
        test_and_animate(model, dataset, ep)
    print(f"Epoch {ep+1}, Loss: {avg_loss:.6f}")
