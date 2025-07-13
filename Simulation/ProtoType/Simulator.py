import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
L = 1.0           # Length of the box (square domain)
N = 128            # Number of grid points per axis
dx = L / N        # Grid spacing
dt = 0.0005       # Time step
T = 5.0           # Total simulation time
steps = int(T / dt)

# Physical constants (natural units)
hbar = 1.0
m = 1.0
alpha = 5.0       # Interaction strength (positive: repulsion, negative: attraction)

# Create spatial grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial wavefunctions: stationary states ψ_1,1 and ψ_2,1
def psi_nm(n, m, X, Y, L):
    return np.sqrt(4 / L**2) * np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L)
def gaussian_packet(X, Y, x0, y0, sigma, kx=0.0, ky=0.0):
    """
    Creates a normalized 2D Gaussian wave packet with optional momentum.
    """
    envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    phase = np.exp(1j * (kx * X + ky * Y))  # momentum via plane wave
    packet = envelope * phase
    norm = np.sqrt(np.sum(np.abs(packet)**2) * dx**2)
    return packet / norm

psi_A = gaussian_packet(X, Y, x0=0.3, y0=0.5, sigma=0.05, kx=0.0, ky=-0.0)  # moves right
psi_B = gaussian_packet(X, Y, x0=0.7, y0=0.5, sigma=0.1, kx=-0.0, ky=0.0) # moves left

# Laplacian operator using finite differences
def laplacian_dirichlet(psi, dx):
    padded = np.pad(psi, pad_width=1, mode='constant', constant_values=0)
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
    ) / dx**2
    return lap

# Time evolution
frames = []
for t in range(steps):
    lap_A = laplacian_dirichlet(psi_A, dx)
    lap_B = laplacian_dirichlet(psi_B, dx)

    # Compute probability densities
    rho_A = np.abs(psi_A)**2
    rho_B = np.abs(psi_B)**2

    # Schrödinger update (Euler method)
    dpsi_A = (1j * hbar / (2 * m)) * lap_A - 1j * alpha * rho_B * psi_A
    dpsi_B = (1j * hbar / (2 * m)) * lap_B - 1j * alpha * rho_A * psi_B

    psi_A += dt * dpsi_A
    psi_B += dt * dpsi_B

    # Normalize wavefunctions
    norm_A = np.sqrt(np.sum(np.abs(psi_A)**2) * dx**2)
    norm_B = np.sqrt(np.sum(np.abs(psi_B)**2) * dx**2)
    psi_A /= norm_A
    psi_B /= norm_B

    # Store frame every 100 steps
    if t % 20 == 0:
        frames.append((np.abs(psi_A.copy())**2, np.abs(psi_B.copy())**2))
    total_prob_A = np.sum(np.abs(psi_A)**2) * dx**2
    total_prob_B = np.sum(np.abs(psi_B)**2) * dx**2
    print(f"t={t*dt:.3f}, norm_A={total_prob_A:.6f}, norm_B={total_prob_B:.6f}")

plt.imshow(np.abs(psi_A)**2, cmap='gray')
plt.title("Final |ψ_A|²")
plt.colorbar()
plt.waitforbuttonpress()


plt.imshow(np.abs(psi_B)**2, cmap='gray')
plt.title("Final |ψ_B|²")
plt.colorbar()
plt.waitforbuttonpress()


print("displayed plot")
frames_array = np.array(frames)
frames_array.shape  # (num_frames, 2, N, N) -> (time, [A, B], x, y)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Prepare the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
cmap = 'Greys'

# Initial plots
im1 = ax1.imshow(frames_array[0, 0], cmap=cmap, origin='lower', extent=[0, L, 0, L])
im2 = ax2.imshow(frames_array[0, 1], cmap=cmap, origin='lower', extent=[0, L, 0, L])
ax1.set_title("Particle A |ψ|²")
ax2.set_title("Particle B |ψ|²")

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

# Animation update function
def update(frame):
    im1.set_data(frames_array[frame, 0])
    im2.set_data(frames_array[frame, 1])
    ax1.set_xlabel(f'Time step: {frame * 100}')
    ax2.set_xlabel(f'Time step: {frame * 100}')
    return [im1, im2]

# Create animation
ani = FuncAnimation(fig, update, frames=len(frames_array), interval=200, blit=True)

# Save the animation as a GIF file
ani.save('quantum_particles.gif', writer=PillowWriter(fps=25))

# plt.tight_layout()
# # plt.show()

# # Save the animation as an MP4 video file
# from matplotlib.animation import FFMpegWriter

# video_path = "Int.mp4"
# writer = FFMpegWriter(fps=5, bitrate=1800)
# ani.save(video_path, writer=writer)

# video_path
