import torch
import h5py
import numpy as np
from tqdm import tqdm

# Simulation constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
L = 1.0
N = 64
dx = L / N
dt = 0.0005
T = 0.5
steps = int(T / dt)
save_interval = 200
sigma = 0.05
num_samples = 100

# Grid
x = torch.linspace(0, L, N, device=device)
y = torch.linspace(0, L, N, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Gaussian wave packet generator
def gaussian_packet(x0, y0, kx, ky):
    envelope = torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    phase = torch.exp(1j * (kx * X + ky * Y))
    packet = envelope * phase
    norm = torch.sqrt(torch.sum(torch.abs(packet)**2) * dx**2)
    return packet / norm

# Laplacian with Dirichlet boundaries
def laplacian(psi):
    padded = torch.nn.functional.pad(psi, (1, 1, 1, 1), mode='constant', value=0)
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
    ) / dx**2
    return lap

# Simulate and return wavefunction evolution
def run_simulation(x0_A, y0_A, kx_A, ky_A, x0_B, y0_B, kx_B, ky_B, alpha):
    psi_A = gaussian_packet(x0_A, y0_A, kx_A, ky_A)
    psi_B = gaussian_packet(x0_B, y0_B, kx_B, ky_B)
    evolution = []

    for t in range(steps):
        lap_A = laplacian(psi_A)
        lap_B = laplacian(psi_B)
        rho_A = torch.abs(psi_A)**2
        rho_B = torch.abs(psi_B)**2

        dpsi_A = (1j / 2.0) * lap_A - 1j * alpha * rho_B * psi_A
        dpsi_B = (1j / 2.0) * lap_B - 1j * alpha * rho_A * psi_B

        psi_A += dt * dpsi_A
        psi_B += dt * dpsi_B

        # Reflective boundaries
        psi_A[0, :] = psi_A[-1, :] = psi_A[:, 0] = psi_A[:, -1] = 0
        psi_B[0, :] = psi_B[-1, :] = psi_B[:, 0] = psi_B[:, -1] = 0

        # Normalize
        psi_A /= torch.sqrt(torch.sum(torch.abs(psi_A)**2) * dx**2)
        psi_B /= torch.sqrt(torch.sum(torch.abs(psi_B)**2) * dx**2)

        if t % save_interval == 0:
            frame_A = torch.stack([psi_A.real, psi_A.imag])
            frame_B = torch.stack([psi_B.real, psi_B.imag])
            evolution.append(torch.stack([frame_A, frame_B]).cpu())

    return torch.stack(evolution)  # (frames, 2 particles, 2 channels, N, N)

# Save simulations
with h5py.File("quantum_wavefunctions_complex.h5", 'w') as f:
    for i in tqdm(range(num_samples), desc="Generating wavefunctions"):
        x0_A = torch.rand(1).item() * 0.4 + 0.1
        y0_A = torch.rand(1).item() * 0.8 + 0.1
        x0_B = torch.rand(1).item() * 0.4 + 0.5
        y0_B = torch.rand(1).item() * 0.8 + 0.1
        kx_A = torch.randn(1).item() * 20
        ky_A = torch.randn(1).item() * 20
        kx_B = torch.randn(1).item() * 20
        ky_B = torch.randn(1).item() * 20
        alpha = torch.rand(1).item() * 10 - 5

        wavefunction = run_simulation(x0_A, y0_A, kx_A, ky_A, x0_B, y0_B, kx_B, ky_B, alpha)
        grp = f.create_group(f"sample_{i}")
        grp.create_dataset("wavefunction", data=wavefunction.numpy())
        grp.create_dataset("params", data=np.array([x0_A, y0_A, kx_A, ky_A, x0_B, y0_B, kx_B, ky_B, alpha]))
