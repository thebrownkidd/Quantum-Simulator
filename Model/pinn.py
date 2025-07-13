import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from torch.autograd import grad

# ==== Data Generation ====
def generate_wave_data(batch_size=1, T=10, N=32, dt=0.001, L=1.0, save_path='wave_data_ann.h5'):
    
    print("Generating data (without Coulomb interaction)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hbar = 1.0
    m = 1.0
    dx = L / N
    x = torch.linspace(0, L, N, device=device)
    y = torch.linspace(0, L, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    def psi_nm(n, m):
        return torch.sqrt(torch.tensor(4.0, device=device)) * torch.sin(n * np.pi * X) * torch.sin(m * np.pi * Y)

    def laplacian(psi):
        real = F.pad(psi.real.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect').squeeze()
        imag = F.pad(psi.imag.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect').squeeze()
        padded = real + 1j * imag
        center = padded[1:-1, 1:-1]
        up     = padded[ :-2, 1:-1]
        down   = padded[2:  , 1:-1]
        left   = padded[1:-1,  :-2]
        right  = padded[1:-1, 2:  ]
        return (up + down + left + right - 4 * center) / dx**2

    all_data = []
    for b in range(batch_size):
        psi_A = psi_nm(1, 1).to(torch.complex128)
        psi_B = psi_nm(2, 1).to(torch.complex128)
        frames = []

        for _ in range(T):
            lap_A = laplacian(psi_A)
            lap_B = laplacian(psi_B)
            dpsi_A = (1j * hbar / (2 * m)) * lap_A
            dpsi_B = (1j * hbar / (2 * m)) * lap_B
            psi_A = psi_A + dt * dpsi_A
            psi_B = psi_B + dt * dpsi_B

            norm_A = torch.sqrt(torch.sum(psi_A.abs()**2) * dx**2)
            norm_B = torch.sqrt(torch.sum(psi_B.abs()**2) * dx**2)
            psi_A = psi_A / norm_A
            psi_B = psi_B / norm_B

            frame = torch.stack([
                psi_A.real, psi_A.imag,
                psi_B.real, psi_B.imag
            ], dim=0)
            frames.append(frame.cpu())

        sim_tensor = torch.stack(frames, dim=0).numpy().astype(np.float32)
        all_data.append(sim_tensor)

    print("Saving data to", save_path)
    with h5py.File(save_path, 'w') as f:
        for i, wave in enumerate(all_data):
            f.create_dataset(f'sample_{i}', data=wave)
    print("Data generation complete.")

# ==== Dataset ====
class WaveDataset(Dataset):
    def __init__(self, path, T=10):
        self.data = []
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                waves = f[key][:]  # [T, 4, N, N]
                for t in range(T - 1):
                    self.data.append((waves[t], waves[t + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# ==== ANN with Flatten ====
class FlattenANN(nn.Module):
    def __init__(self, input_dim=4*32*32, hidden_dim=1024, output_dim=4*32*32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Unflatten(1, (4, 32, 32))
        )

    def forward(self, x):
        return self.model(x)

# ==== Schrödinger Residual Loss ====
def schrodinger_residual(pred, input_wave, dt, dx):
    pred.requires_grad_()
    ψA = pred[:, 0] + 1j * pred[:, 1]
    ψB = pred[:, 2] + 1j * pred[:, 3]
    ψA_prev = input_wave[:, 0] + 1j * input_wave[:, 1]
    ψB_prev = input_wave[:, 2] + 1j * input_wave[:, 3]

    laplace = lambda u: (
        -4 * u +
        torch.roll(u, 1, -1) + torch.roll(u, -1, -1) +
        torch.roll(u, 1, -2) + torch.roll(u, -1, -2)
    ) / dx**2

    lap_A = laplace(ψA)
    lap_B = laplace(ψB)
    dψA_dt = (ψA - ψA_prev) / dt
    dψB_dt = (ψB - ψB_prev) / dt

    res_A = (1j * dψA_dt + 0.5 * lap_A).abs().mean()
    res_B = (1j * dψB_dt + 0.5 * lap_B).abs().mean()
    return res_A + res_B

# ==== Reflective Boundary Condition Loss ====
def reflective_boundary_condition_loss(output):
    left = output[:, :, :, 0]
    right = output[:, :, :, -1]
    top = output[:, :, 0, :]
    bottom = output[:, :, -1, :]
    return (left**2 + right**2 + top**2 + bottom**2).mean()

def visualize(pred, target, epoch ,save_path="actualvpred.png"):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    titles = ["ψA Real", "ψA Imag", "ψB Real", "ψB Imag"]
    for i in range(4):
        axs[0, i].imshow(target[0, i].detach().cpu().numpy(), cmap='viridis')
        axs[0, i].set_title(f"True {titles[i]}")
        axs[0, i].axis('off')

        axs[1, i].imshow(pred[0, i].detach().cpu().numpy(), cmap='viridis')
        axs[1, i].set_title(f"Pred {titles[i]}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(("plots/"+str(epoch) +save_path))
    plt.close()
    
# ==== Training ====
def train_ann():
    losses = []
    per_errs = []
    eps = []
    generate_wave_data(batch_size=5)
    dataset = WaveDataset("wave_data_ann.h5")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = FlattenANN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(600):
        total_loss = 0
        pe = 0
        i = 0
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            mse = loss_fn(pred, y)
            pde = schrodinger_residual(pred, x, dt=0.001, dx=1/64)
            bc = reflective_boundary_condition_loss(pred)
            loss = 10000*mse + 0.0001*pde + bc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pe += 100*(abs(pred - y)).mean() / abs(y).mean()
            per_errs.append((100*(abs(pred - y)).mean() / abs(y).mean()).item())
            losses.append(loss.item())
            eps.append(epoch + (i+1)/len(dataloader))
            i +=1
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.6f}, PDE: {pde}, BC: {bc}, MSE: {mse}, PercERR: {pe/len(dataloader)}")
        if (epoch+1)%30 == 0:
            visualize(pred, y, epoch)

    plt.scatter(eps, losses, s = 0.1, color = 'black')
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.savefig("losses.png", dpi = 200)
    plt.cla()
    
    
    plt.scatter(eps, per_errs,s = 0.1, color = 'black')
    plt.xlabel("Epochs")
    plt.ylabel("Error (%)")
    plt.savefig("error.png", dpi = 200)
    plt.cla()


if __name__ == '__main__':
    train_ann()
