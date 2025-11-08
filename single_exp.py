# %%
# Experiment: Diagonal complex SSM (LRU-style parameterization) trained on shift task
# Author: ChatGPT for Yuval Ran-Milo (@yuv.milo)
# 
# This file is organized in Jupyter-style cells (compatible with Cursor / VSCode).
# The first cell defines hyperparameters. Subsequent cells define the model, data generation,
# training loop, plotting utilities, and a one-call `run_experiment` convenience function.
#
# Notes:
# - Recurrent matrix A is complex diagonal with entries lambda_i = r_i * exp(i * theta_i).
# - Radii are parameterized in a stable LRU-like way: r = exp(-softplus(rho)) \in (0, 1).
#   (See Orvieto et al. 2023 "LRU": stable exponential parameterization.)
# - Output is the real part of a complex readout: y_t = Re(c^H x_t). We keep B, C fixed by default
#   to isolate the dynamics of A during training.
# - Input per batch is a linear combination of chosen frequencies with Gaussian coefficients,
#   normalized to have L2 norm 1.
# - Target is the input shifted to the right by SHIFT_SIZE (zeros padded on the left).
# - We save a figure with the trajectory of diag(A) eigenvalues on the complex plane
#   across training (unit circle shown), and a JSON of all hyperparameters + training logs.
#
# Requirements: PyTorch, matplotlib, numpy. (Available in the execution environment.)
# %%

# ---------------------------
# Hyperparameters (edit here)
# ---------------------------

SEED = 3
DEVICE = "cpu"

# Experiment management
CLEAR_EXP = True  # Set to True to clear all previous experiments on startup

# Model architecture flags
BC_REAL = False          # Set to True to constrain B and C to be real-valued
B_IS_ONE = True         # Set to True to fix B to ones (not learnable)
C_IS_ONE = False        # Set to True to fix C to ones (not learnable)
TAKE_REAL = False       # Set to True to take real part of output; False for complex output and targets

# SSM size and sequence/batch config
N = 1                    # dimension of the SSM (number of eigenvalues on diagonal)
SEQ_LEN = 64            # sequence length per batch
BATCH_SIZE = 512          # number of sequences per batch
SHIFT_SIZE = 0           # how many steps to shift target to the right

# Training config
NUM_STEPS = 500          # gradient steps
OPTIMIZER = "adam"       # "adam" or "sgd"
LR_ADAM = 5e-3
LR_SGD = 1e-2
WEIGHT_DECAY = 0.0       # optional
NOISE_LEVEL = 0.0        # std of noise to add to inputs during training (0 = no noise)

LOG_EVERY = 10           # record eigenvalues and loss every k steps

# Frequencies to synthesize inputs (normalized to [0, 1] where 1 = π radians)
# e.g., 0.1 means 0.1*π radians, 1.0 means π radians
TARGET_FREQUENCIES = [0.15,]  # list of normalized frequencies (0 to 1)

# Initial radii and angles for A (length N)
# Provide lists; radii in (0,1), angles in (-pi, pi]
import math
INIT_RADII  = [0.9] * N
INIT_ANGLES = [ -math.pi, ]  # spread around unit circle

# Logging / saving
BASE_PATH = "./ssm_experiments_single"  # results root
RUN_NAME_PREFIX = "shift_task"

# ---------------------------
# End hyperparameters section
# ---------------------------

# %%
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set seeds
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)
torch.set_default_dtype(torch.float32)

# %%
# Experiment cleanup utility

def clear_experiments(base_path: str = BASE_PATH):
    """
    Clear all previous experiment results.
    """
    import shutil
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Cleared all experiments in {base_path}")
    else:
        print(f"No experiments found at {base_path}")

# Clear experiments if requested
if CLEAR_EXP:
    clear_experiments()

# %%
# Utility: parameterizations and helpers

def stable_radius_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """
    LRU-like stable exponential parameterization:
    r = exp(-softplus(rho)) ∈ (0,1). 
    This ensures magnitudes are strictly inside the unit circle.
    """
    return torch.exp(-F.softplus(rho))

def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return (theta + math.pi) % (2*math.pi) - math.pi

def complex_diag_from_rt(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Return complex diagonal vector λ = r * exp(iθ)."""
    return r * torch.exp(1j * theta)

# %%
# Data generation

def sample_frequency_mixture(freqs: List[float], seq_len: int, batch_size: int, 
                            take_real: bool = True) -> torch.Tensor:
    """
    Create a batch of sequences (batch, seq_len) where each sequence is a
    linear combination over provided `freqs` with Gaussian coefficients.
    
    Args:
        freqs: List of normalized frequencies in [0, 1] where 1 corresponds to π radians
        seq_len: Length of each sequence
        batch_size: Number of sequences in batch
        take_real: If True, use cos(w t) (real); if False, use exp(i*w*t) (complex)
    """
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # (1, T)
    # Convert normalized frequencies (0-1) to radians (0-π)
    Fm = torch.tensor(freqs, dtype=torch.float32).view(-1, 1) * math.pi  # (M, 1)
    
    if take_real:
        # Real-valued: use cosines
        basis_bank = torch.cos(Fm * t)                           # (M, T)
        coeffs = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        mix = coeffs @ basis_bank                                # (batch, T)
    else:
        # Complex-valued: use exp(i*w*t)
        basis_bank = torch.exp(1j * Fm * t)                      # (M, T) complex
        # Complex Gaussian coefficients
        coeffs_real = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        coeffs_imag = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        coeffs = torch.complex(coeffs_real, coeffs_imag)
        mix = coeffs @ basis_bank                                # (batch, T) complex

    # Normalize each sequence to norm 1
    norms = torch.linalg.norm(mix, ord=2, dim=1, keepdim=True) + 1e-12
    #mix = mix / norms
    return mix  # (batch, T)

def make_shift_targets(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shift each sequence in x (batch, T) to the right by `shift` with zero-padding.
    y[t] = x[t - shift] for t >= shift, else 0.
    """
    b, T = x.shape
    y = torch.zeros_like(x)
    if shift <= 0:
        return x
    y[:, shift:] = x[:, :-shift]
    return y

# %%
# Model

class DiagonalSSM(nn.Module):
    """
    Complex diagonal State-Space / linear RNN:
        x_{t+1} = diag(λ) x_t + b * u_t
        y_t     = Re( c^H x_t ) or c^H x_t (depends on take_real flag)
    Trainable parameters: (ρ, θ) -> λ = exp(-softplus(ρ)) * exp(i θ), and optionally b, c
    """
    def __init__(self, init_radii: List[float], init_angles: List[float], 
                 learn_c: bool=True, bc_real: bool=False, b_is_one: bool=True, 
                 c_is_one: bool=False, take_real: bool=True):
        super().__init__()
        assert len(init_radii) == len(init_angles), "radii and angles lengths must match"
        n = len(init_radii)
        self.n = n
        self.bc_real = bc_real
        self.b_is_one = b_is_one
        self.c_is_one = c_is_one
        self.take_real = take_real

        # Raw parameters:
        # initialize ρ from r via inverse: r = exp(-softplus(ρ)) → choose ρ ≈ log(expm1(-log r))
        r = torch.tensor(init_radii, dtype=torch.float32)
        r = torch.clamp(r, 1e-6, 1-1e-6)
        rho0 = torch.log(torch.expm1(-torch.log(r)))  # invert: softplus(rho) = -log r
        theta0 = torch.tensor(init_angles, dtype=torch.float32)

        self.rho   = nn.Parameter(rho0)     # unconstrained (maps to r in (0,1))
        self.theta = nn.Parameter(theta0)   # (will be wrapped for reporting, but unconstrained for smoothness)

        # B and C initialization based on flags
        dtype = torch.float32 if bc_real else torch.complex64
        self.learn_c = learn_c
        
        # Initialize B
        if b_is_one:
            # B is fixed to ones (not learnable)
            self.b = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=False)
        else:
            # B is learnable, initialized to ones
            self.b = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=True)
        
        # Initialize C (always initialized to ones)
        if c_is_one:
            # C is fixed to ones (not learnable)
            self.c = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=False)
        else:
            # C's learnability depends on learn_c flag
            self.c = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=learn_c)

    def eigenvalues(self) -> torch.Tensor:
        """Return λ (complex) as vector shape (n,)"""
        r = stable_radius_from_rho(self.rho)
        lam = complex_diag_from_rt(r.to(self.theta.device), self.theta)
        return lam.to(torch.complex64)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : (batch, T) real or complex depending on take_real flag
        returns y : (batch, T) real if take_real=True, complex otherwise
        """
        batch, T = u.shape
        lam = self.eigenvalues()             # (n,)
        x = torch.zeros(batch, self.n, dtype=torch.complex64, device=u.device)
        
        # Determine output dtype
        output_dtype = torch.float32 if self.take_real else torch.complex64
        y = torch.zeros(batch, T, dtype=output_dtype, device=u.device)

        # Broadcast b, c (convert to complex if they are real, preserving gradients)
        b = self.b.to(x.device)  # (n,)
        c = self.c.to(x.device)  # (n,)
        if self.bc_real:
            # Properly convert real to complex preserving gradients
            b = torch.complex(b, torch.zeros_like(b))
            c = torch.complex(c, torch.zeros_like(c))

        for t in range(T):
            # Handle complex or real input
            u_t = u[:, t].unsqueeze(-1)  # (batch, 1)
            if not torch.is_complex(u_t):
                u_t = torch.complex(u_t, torch.zeros_like(u_t))
            
            x = lam * x + b * u_t     # (batch, n)
            y_t = torch.matmul(x, torch.conj(c))  # (batch,)
            
            if self.take_real:
                y[:, t] = torch.real(y_t)
            else:
                y[:, t] = y_t

        return y

# %%
# Training and logging utilities

def complex_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss for both real and complex tensors.
    For complex tensors: MSE = mean(|output - target|^2)
    For real tensors: standard MSE
    """
    if torch.is_complex(output) or torch.is_complex(target):
        # For complex: mean of squared magnitudes of differences
        diff = output - target
        return torch.mean(torch.real(diff * torch.conj(diff)))
    else:
        # For real: standard MSE
        return F.mse_loss(output, target)

@dataclass
class TrainLogs:
    steps: List[int]
    losses: List[float]
    lambdas: List[List[complex]]  # list over checkpoints, each is list of n complex numbers
    c_values: List[List[complex]]  # list over checkpoints, each is list of n complex numbers

def train_shift_task(
    model: DiagonalSSM,
    freqs: List[float],
    seq_len: int,
    batch_size: int,
    shift_size: int,
    num_steps: int,
    optimizer_name: str = "adam",
    lr_adam: float = 5e-3,
    lr_sgd: float = 1e-2,
    weight_decay: float = 0.0,
    log_every: int = 10,
    device: str = "cpu",
    loss_threshold: float = 1e-5,
    max_steps: int = 50000,
    take_real: bool = True,
    noise_level: float = 0.0,
) -> TrainLogs:
    model.to(device)
    model.train()

    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.0, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    steps, losses, lambdas, c_values = [], [], [], []

    # Record initial eigenvalues and C values
    with torch.no_grad():
        lam0 = model.eigenvalues().detach().cpu().numpy().tolist()
        c0 = model.c.detach().cpu().numpy().tolist()
    lambdas.append([complex(z) for z in lam0])
    c_values.append([complex(z) for z in c0])
    steps.append(0)
    losses.append(float("nan"))

    # Use tqdm for progress bar
    pbar = tqdm(range(1, max_steps + 1), desc="Training", unit="step")
    
    for step in pbar:
        u = sample_frequency_mixture(freqs, seq_len, batch_size, take_real=take_real).to(device)  # (B, T)
        
        # Add noise to input if noise_level > 0
        if noise_level > 0:
            if torch.is_complex(u):
                # Complex noise: Gaussian noise in both real and imaginary parts
                noise = noise_level * torch.complex(
                    torch.randn_like(u.real, device=device),
                    torch.randn_like(u.imag, device=device)
                )
            else:
                # Real noise
                noise = noise_level * torch.randn_like(u, device=device)
            u_noisy = u + noise
        else:
            u_noisy = u
        
        # Target is the shifted noisy input (important: shift the noisy version!)
        y_target = make_shift_targets(u_noisy, shift_size).to(device)  # (B, T)

        y_hat = model(u_noisy)  # (B, T)
        # Use custom MSE loss that handles both real and complex tensors
        loss = complex_mse_loss(y_hat, y_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        current_loss = float(loss.item())
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{current_loss:.2e}", "threshold": f"{loss_threshold:.2e}"})

        if (step % log_every) == 0 or (step == max_steps):
            with torch.no_grad():
                lam = model.eigenvalues().detach().cpu().numpy().tolist()
                c_val = model.c.detach().cpu().numpy().tolist()
                lambdas.append([complex(z) for z in lam])
                c_values.append([complex(z) for z in c_val])
            steps.append(step)
            losses.append(current_loss)
        
        # Check if we've reached the loss threshold
        if current_loss < loss_threshold:
            # Log final values
            with torch.no_grad():
                lam = model.eigenvalues().detach().cpu().numpy().tolist()
                c_val = model.c.detach().cpu().numpy().tolist()
                lambdas.append([complex(z) for z in lam])
                c_values.append([complex(z) for z in c_val])
            steps.append(step)
            losses.append(current_loss)
            pbar.set_postfix({"loss": f"{current_loss:.2e}", "status": "converged!"})
            pbar.close()
            print(f"\nConverged at step {step} with loss {current_loss:.2e}")
            break
    else:
        pbar.close()
        print(f"\nReached max steps ({max_steps}) with final loss {current_loss:.2e}")

    return TrainLogs(steps=steps, losses=losses, lambdas=lambdas, c_values=c_values)

# %%
# Plotting and saving

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_loss_curve(
    steps: List[int],
    losses: List[float],
    save_path: str,
    title: str = "",
    loss_threshold: float = 1e-5,
):
    """
    Plot the loss curve over training.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter out NaN values (e.g., initial step)
    valid_idx = [i for i, loss in enumerate(losses) if not np.isnan(loss)]
    valid_steps = [steps[i] for i in valid_idx]
    valid_losses = [losses[i] for i in valid_idx]
    
    if len(valid_losses) > 0:
        ax.semilogy(valid_steps, valid_losses, linewidth=2, marker='o', markersize=4, label='Training loss')
        
        # Add threshold line
        ax.axhline(y=loss_threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Threshold ({loss_threshold:.0e})', alpha=0.7)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss (MSE, log scale)", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='best', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def plot_c_and_radii_vs_time(
    steps: List[int],
    c_checkpoints: List[List[complex]],
    lambdas_checkpoints: List[List[complex]],
    save_path: str,
    title: str = "",
    target_frequencies: List[float] = None,
):
    """
    Plot C values and radii as functions of training steps.
    - steps: list of training step numbers
    - c_checkpoints: list over checkpoints, each is a list of n complex numbers
    - lambdas_checkpoints: list over checkpoints, each is a list of n complex numbers (to extract radii)
    - target_frequencies: list of normalized frequencies (for computing 1-r*cos(delta))
    """
    checkpoints = len(c_checkpoints)
    n = len(c_checkpoints[0])

    # Prepare arrays
    c_array = np.array(c_checkpoints, dtype=np.complex64)  # (C, n)
    lam_array = np.array(lambdas_checkpoints, dtype=np.complex64)  # (C, n)
    
    # Extract C values (real part if BC_REAL=True, magnitude if complex)
    c_vals = np.real(c_array)  # (C, n) - real part of C
    
    # Extract radii from eigenvalues
    radii = np.abs(lam_array)  # (C, n)
    
    # Extract angles from eigenvalues
    angles = np.angle(lam_array)  # (C, n) in radians

    # Determine if we should add the 1-r*cos(delta) metric (when 1 input frequency)
    add_delta_metric = (target_frequencies is not None and len(target_frequencies) == 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot C values
    for j in range(n):
        ax1.plot(steps, c_vals[:, j], linewidth=2, marker='o', markersize=4, 
                label=f'c_{j+1}', alpha=0.8)
    
    # Add 1 - r*cos(delta) on the same plot as C if applicable
    if add_delta_metric:
        # Convert input frequency to radians
        input_freq_rad = target_frequencies[0] * np.pi
        
        # Plot metric for each eigenvalue
        for j in range(n):
            # Compute delta: angular distance between input frequency and eigenvalue angle
            delta = np.abs(input_freq_rad - angles[:, j])  # (C,)
            
            # Wrap delta to [0, π] (shortest angular distance)
            delta = np.minimum(delta, 2*np.pi - delta)
            
            # Compute 1 - r*cos(delta)
            metric = 1 - radii[:, j] * np.cos(delta)  # (C,)
            
            ax1.plot(steps, metric, linewidth=2, marker='d', markersize=4, 
                    linestyle='--', label=f'1-r_{j+1}·cos(δ_{j+1})', alpha=0.8)
        
        ax1.axhline(y=0.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Optimal (δ metric)')
    
    ax1.set_ylabel("C values & 1-r·cos(δ)", fontsize=12)
    ax1.set_title(title, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc='best', fontsize=9)
    
    # Plot radii
    for j in range(n):
        ax2.plot(steps, radii[:, j], linewidth=2, marker='s', markersize=4, 
                label=f'r_{j+1}', alpha=0.8)
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.3, label='Unit circle')
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Radii (|λ|)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_ylim([0, 1.1])  # Radii should be in (0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def eigen_trajectory_plot(
    lambdas_checkpoints: List[List[complex]],
    save_path: str,
    title: str = "",
):
    """
    Plot eigenvalue trajectories in the complex plane.
    - lambdas_checkpoints: list over checkpoints, each is a list of n complex numbers.
    """
    checkpoints = len(lambdas_checkpoints)
    n = len(lambdas_checkpoints[0])

    # Prepare arrays of shape (checkpoints, n)
    traj = np.array(lambdas_checkpoints, dtype=np.complex64)  # (C, n)

    # plot
    fig, ax = plt.subplots(figsize=(7, 7))
    # unit circle
    thetas = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(thetas), np.sin(thetas), 'k--', linewidth=1.0, alpha=0.3, label='Unit circle')

    # Use a colormap to show time progression
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, checkpoints))

    # trajectories
    for j in range(n):
        real_vals = traj[:, j].real
        imag_vals = traj[:, j].imag
        
        # Plot trajectory with color gradient
        for i in range(checkpoints - 1):
            ax.plot(real_vals[i:i+2], imag_vals[i:i+2], 
                   color=colors[i], linewidth=2.0, alpha=0.7)
        
        # Mark initial position (larger marker)
        ax.plot(real_vals[0], imag_vals[0], 'o', 
               color=colors[0], markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'λ_{j+1} start' if j < 3 else None, zorder=5)
        
        # Mark final position (star marker)
        ax.plot(real_vals[-1], imag_vals[-1], '*', 
               color=colors[-1], markersize=15,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'λ_{j+1} end' if j < 3 else None, zorder=5)
        
        # Add arrow to show direction (at midpoint)
        if checkpoints > 3:
            mid_idx = checkpoints // 2
            dx = real_vals[mid_idx + 1] - real_vals[mid_idx]
            dy = imag_vals[mid_idx + 1] - imag_vals[mid_idx]
            ax.arrow(real_vals[mid_idx], imag_vals[mid_idx], 
                    dx * 0.5, dy * 0.5,
                    head_width=0.03, head_length=0.02, 
                    fc=colors[mid_idx], ec=colors[mid_idx], 
                    linewidth=1.5, alpha=0.8, zorder=4)

    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=checkpoints-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Training step checkpoint')
    
    ax.set_xlabel("Real", fontsize=12)
    ax.set_ylabel("Imag", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def make_run_dir(
    base_path: str,
    run_name_prefix: str,
    init_radii: List[float],
    init_angles: List[float],
    target_freqs: List[float],
) -> str:
    """
    Make a human-readable directory name with key info.
    """
    def fmt_list(xs, fmt="{:.3f}", k=5):
        xs = list(xs)
        tail = "" if len(xs) <= k else f"_plus{len(xs)-k}"
        return "_".join(fmt.format(v) for v in xs[:k]) + tail

    ts = time.strftime("%Y%m%d-%H%M%S")
    dir_name = (
        f"{run_name_prefix}"
        f"__n{len(init_radii)}__shift{SHIFT_SIZE}"
        f"__freqs_{fmt_list(target_freqs)}"
        f"__r0_{fmt_list(init_radii)}"
        f"__th0_{fmt_list(init_angles)}"
        f"__{ts}"
    )
    run_dir = os.path.join(base_path, dir_name)
    _ensure_dir(run_dir)
    return run_dir

def plot_example_io(
    input_seq: np.ndarray,
    output_seq: np.ndarray, 
    target_seq: np.ndarray,
    save_path: str,
    title: str = "",
):
    """
    Plot a single example of input, output, and target sequences.
    Handles both real and complex data.
    """
    is_complex = np.iscomplexobj(input_seq) or np.iscomplexobj(output_seq) or np.iscomplexobj(target_seq)
    
    if is_complex:
        # Plot real and imaginary parts separately
        fig, axes = plt.subplots(3, 2, figsize=(14, 8))
        time_steps = np.arange(len(input_seq))
        
        # Input - Real
        axes[0, 0].plot(time_steps, np.real(input_seq), linewidth=2, color='blue', label='Input (real)')
        axes[0, 0].set_ylabel("Input (real)", fontsize=12)
        axes[0, 0].grid(True, linestyle="--", alpha=0.4)
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].set_title(title, fontsize=10)
        
        # Input - Imaginary
        axes[0, 1].plot(time_steps, np.imag(input_seq), linewidth=2, color='cyan', label='Input (imag)')
        axes[0, 1].set_ylabel("Input (imag)", fontsize=12)
        axes[0, 1].grid(True, linestyle="--", alpha=0.4)
        axes[0, 1].legend(loc='upper right')
        
        # Target - Real
        axes[1, 0].plot(time_steps, np.real(target_seq), linewidth=2, color='green', label='Target (real)')
        axes[1, 0].set_ylabel("Target (real)", fontsize=12)
        axes[1, 0].grid(True, linestyle="--", alpha=0.4)
        axes[1, 0].legend(loc='upper right')
        
        # Target - Imaginary
        axes[1, 1].plot(time_steps, np.imag(target_seq), linewidth=2, color='lightgreen', label='Target (imag)')
        axes[1, 1].set_ylabel("Target (imag)", fontsize=12)
        axes[1, 1].grid(True, linestyle="--", alpha=0.4)
        axes[1, 1].legend(loc='upper right')
        
        # Output - Real
        axes[2, 0].plot(time_steps, np.real(output_seq), linewidth=2, color='red', label='Output (real)')
        axes[2, 0].set_ylabel("Output (real)", fontsize=12)
        axes[2, 0].set_xlabel("Time Step", fontsize=12)
        axes[2, 0].grid(True, linestyle="--", alpha=0.4)
        axes[2, 0].legend(loc='upper right')
        
        # Output - Imaginary
        axes[2, 1].plot(time_steps, np.imag(output_seq), linewidth=2, color='orange', label='Output (imag)')
        axes[2, 1].set_ylabel("Output (imag)", fontsize=12)
        axes[2, 1].set_xlabel("Time Step", fontsize=12)
        axes[2, 1].grid(True, linestyle="--", alpha=0.4)
        axes[2, 1].legend(loc='upper right')
    else:
        # Original real-valued plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        time_steps = np.arange(len(input_seq))
        
        # Input
        axes[0].plot(time_steps, input_seq, linewidth=2, color='blue', label='Input')
        axes[0].set_ylabel("Input", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(loc='upper right')
        axes[0].set_title(title, fontsize=10)
        
        # Target
        axes[1].plot(time_steps, target_seq, linewidth=2, color='green', label='Target')
        axes[1].set_ylabel("Target", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(loc='upper right')
        
        # Output
        axes[2].plot(time_steps, output_seq, linewidth=2, color='red', label='Model Output')
        axes[2].set_ylabel("Output", fontsize=12)
        axes[2].set_xlabel("Time Step", fontsize=12)
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def plot_noisy_example_io(
    noisy_input_seq: np.ndarray,
    output_seq: np.ndarray, 
    clean_target_seq: np.ndarray,
    save_path: str,
    title: str = "",
):
    """
    Plot a single example showing:
    - Input with noise
    - Model output (on noisy input)
    - Target (based on clean input without noise)
    Handles both real and complex data.
    """
    is_complex = np.iscomplexobj(noisy_input_seq) or np.iscomplexobj(output_seq) or np.iscomplexobj(clean_target_seq)
    
    if is_complex:
        # Plot real and imaginary parts separately
        fig, axes = plt.subplots(3, 2, figsize=(14, 8))
        time_steps = np.arange(len(noisy_input_seq))
        
        # Noisy Input - Real
        axes[0, 0].plot(time_steps, np.real(noisy_input_seq), linewidth=2, color='blue', label='Noisy Input (real)')
        axes[0, 0].set_ylabel("Noisy Input (real)", fontsize=12)
        axes[0, 0].grid(True, linestyle="--", alpha=0.4)
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].set_title(title + " [Noise Test]", fontsize=10)
        
        # Noisy Input - Imaginary
        axes[0, 1].plot(time_steps, np.imag(noisy_input_seq), linewidth=2, color='cyan', label='Noisy Input (imag)')
        axes[0, 1].set_ylabel("Noisy Input (imag)", fontsize=12)
        axes[0, 1].grid(True, linestyle="--", alpha=0.4)
        axes[0, 1].legend(loc='upper right')
        
        # Clean Target - Real
        axes[1, 0].plot(time_steps, np.real(clean_target_seq), linewidth=2, color='green', label='Target (real)')
        axes[1, 0].set_ylabel("Target (real)", fontsize=12)
        axes[1, 0].grid(True, linestyle="--", alpha=0.4)
        axes[1, 0].legend(loc='upper right')
        
        # Clean Target - Imaginary
        axes[1, 1].plot(time_steps, np.imag(clean_target_seq), linewidth=2, color='lightgreen', label='Target (imag)')
        axes[1, 1].set_ylabel("Target (imag)", fontsize=12)
        axes[1, 1].grid(True, linestyle="--", alpha=0.4)
        axes[1, 1].legend(loc='upper right')
        
        # Output - Real
        axes[2, 0].plot(time_steps, np.real(output_seq), linewidth=2, color='red', label='Output (real)')
        axes[2, 0].set_ylabel("Output (real)", fontsize=12)
        axes[2, 0].set_xlabel("Time Step", fontsize=12)
        axes[2, 0].grid(True, linestyle="--", alpha=0.4)
        axes[2, 0].legend(loc='upper right')
        
        # Output - Imaginary
        axes[2, 1].plot(time_steps, np.imag(output_seq), linewidth=2, color='orange', label='Output (imag)')
        axes[2, 1].set_ylabel("Output (imag)", fontsize=12)
        axes[2, 1].set_xlabel("Time Step", fontsize=12)
        axes[2, 1].grid(True, linestyle="--", alpha=0.4)
        axes[2, 1].legend(loc='upper right')
    else:
        # Original real-valued plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        time_steps = np.arange(len(noisy_input_seq))
        
        # Noisy Input
        axes[0].plot(time_steps, noisy_input_seq, linewidth=2, color='blue', label='Input (with noise)')
        axes[0].set_ylabel("Noisy Input", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(loc='upper right')
        axes[0].set_title(title + " [Noise Test]", fontsize=10)
        
        # Clean Target
        axes[1].plot(time_steps, clean_target_seq, linewidth=2, color='green', label='Target (from clean input)')
        axes[1].set_ylabel("Clean Target", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(loc='upper right')
        
        # Output on Noisy Input
        axes[2].plot(time_steps, output_seq, linewidth=2, color='red', label='Model Output (on noisy input)')
        axes[2].set_ylabel("Output", fontsize=12)
        axes[2].set_xlabel("Time Step", fontsize=12)
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def plot_impulse_response(
    model: DiagonalSSM,
    seq_len: int,
    save_path: str,
    title: str = "",
    device: str = "cpu",
):
    """
    Plot the impulse response of the model in both time and frequency domain.
    Impulse: input is 1 at t=0, then 0 for all subsequent timesteps.
    """
    model.eval()
    
    # Create impulse input: [1, 0, 0, 0, ...]
    impulse = torch.zeros(1, seq_len, dtype=torch.complex64 if not model.take_real else torch.float32, device=device)
    impulse[0, 0] = 1.0
    
    # Get model response
    with torch.no_grad():
        response = model(impulse)  # (1, T)
    
    # Convert to numpy
    response_np = response[0].cpu().numpy()
    
    # Compute DFT of the impulse response
    response_fft = np.fft.fft(response_np)
    freqs = np.fft.fftfreq(seq_len, d=1.0)  # Normalized frequencies
    
    # Check if response is complex
    is_complex = np.iscomplexobj(response_np)
    
    if is_complex:
        # Plot time domain (real and imaginary) and frequency domain
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        time_steps = np.arange(seq_len)
        
        # Time domain - Real part
        ax1.stem(time_steps, np.real(response_np), linefmt='b-', markerfmt='bo', basefmt='k-', label='Real part')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax1.set_ylabel("Response (real)", fontsize=12)
        ax1.set_title("Time Domain - Real Part", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc='best')
        
        # Time domain - Imaginary part
        ax2.stem(time_steps, np.imag(response_np), linefmt='r-', markerfmt='ro', basefmt='k-', label='Imaginary part')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax2.set_ylabel("Response (imag)", fontsize=12)
        ax2.set_title("Time Domain - Imaginary Part", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc='best')
        
        # Frequency domain - Magnitude
        ax3.plot(freqs[:seq_len//2], np.abs(response_fft[:seq_len//2]), 'g-', linewidth=2, label='|H(f)|')
        ax3.set_xlabel("Normalized Frequency", fontsize=12)
        ax3.set_ylabel("Magnitude", fontsize=12)
        ax3.set_title("Frequency Domain - Magnitude", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc='best')
        
        # Frequency domain - Phase
        phase = np.angle(response_fft[:seq_len//2])
        ax4.plot(freqs[:seq_len//2], phase, 'm-', linewidth=2, label='∠H(f)')
        ax4.set_xlabel("Normalized Frequency", fontsize=12)
        ax4.set_ylabel("Phase (radians)", fontsize=12)
        ax4.set_title("Frequency Domain - Phase", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend(loc='best')
        
        # Add overall title
        fig.suptitle(title + " - Impulse Response", fontsize=12, y=0.95)
        
    else:
        # Plot real-valued response in time and frequency domain
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        time_steps = np.arange(seq_len)
        
        # Time domain
        ax1.stem(time_steps, response_np, linefmt='b-', markerfmt='bo', basefmt='k-', label='Impulse response')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax1.set_xlabel("Time Step", fontsize=12)
        ax1.set_ylabel("Response", fontsize=12)
        ax1.set_title("Time Domain", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc='best')
        
        # Frequency domain - Magnitude (for real signals, show positive frequencies only)
        ax2.plot(freqs[:seq_len//2], np.abs(response_fft[:seq_len//2]), 'g-', linewidth=2, label='|H(f)|')
        ax2.set_xlabel("Normalized Frequency", fontsize=12)
        ax2.set_ylabel("Magnitude", fontsize=12)
        ax2.set_title("Frequency Domain - Magnitude", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc='best')
        
        # Add overall title
        fig.suptitle(title + " - Impulse Response", fontsize=12, y=0.95)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def save_result(
    BASE_PATH: str,
    init_radii: List[float],
    init_angles: List[float],
    target_frequencies: List[float],
    logs: TrainLogs,
    hyperparams: Dict[str, Any],
    run_name_prefix: str = "shift_task",
    model: DiagonalSSM = None,
    example_input: torch.Tensor = None,
    example_target: torch.Tensor = None,
    shift_size: int = 0,
    seq_len: int = 64,
    batch_size: int = 16,
    take_real: bool = True,
) -> str:
    """
    Save:
      - hyperparams + logs as JSON
      - eigenvalue trajectory figure (complex plane)
      - C and radii vs time figure
      - loss curve figure
      - impulse response figure
      - example input/output/target (clean)
      - example input+noise/output/target (noise test)
    Returns the directory path used for saving.
    """
    run_dir = make_run_dir(BASE_PATH, run_name_prefix, init_radii, init_angles, target_frequencies)

    # Title string for the figures
    fig_title = (
        f"{run_name_prefix} | N={len(init_radii)}, SHIFT={hyperparams.get('SHIFT_SIZE')}, "
        f"opt={hyperparams.get('OPTIMIZER')}, final_step={logs.steps[-1]}\n"
        f"freqs={np.round(target_frequencies, 4).tolist()} (×π rad)\n"
        f"init_radii (min/max)=({min(init_radii):.3f}, {max(init_radii):.3f})"
    )

    # Save eigenvalue trajectory figure (complex plane)
    fig_path = os.path.join(run_dir, "eigen_trajectory.png")
    eigen_trajectory_plot(logs.lambdas, save_path=fig_path, title=fig_title)
    
    # Save C and radii vs time figure
    c_fig_path = os.path.join(run_dir, "c_and_radii_vs_time.png")
    plot_c_and_radii_vs_time(logs.steps, logs.c_values, logs.lambdas, save_path=c_fig_path, 
                            title=fig_title, target_frequencies=target_frequencies)
    
    # Save loss curve figure
    loss_fig_path = os.path.join(run_dir, "loss_curve.png")
    loss_threshold = hyperparams.get('LOSS_THRESHOLD', 1e-5)
    plot_loss_curve(logs.steps, logs.losses, save_path=loss_fig_path, 
                   title=fig_title, loss_threshold=loss_threshold)
    
    # Save impulse response if model is provided
    if model is not None:
        impulse_fig_path = os.path.join(run_dir, "impulse_response.png")
        plot_impulse_response(model, seq_len, save_path=impulse_fig_path, 
                            title=fig_title, device=example_input.device if example_input is not None else 'cpu')
    
    # Save example input/output/target if provided
    if model is not None and example_input is not None and example_target is not None:
        model.eval()
        with torch.no_grad():
            example_output = model(example_input)
        
        # Take first sequence from batch
        input_np = example_input[0].cpu().numpy()
        output_np = example_output[0].cpu().numpy()
        target_np = example_target[0].cpu().numpy()
        
        io_fig_path = os.path.join(run_dir, "example_io.png")
        plot_example_io(input_np, output_np, target_np, save_path=io_fig_path, title=fig_title)
        
        # Generate noisy example
        # Create clean input
        clean_input = sample_frequency_mixture(target_frequencies, seq_len, 1, take_real=take_real).to(example_input.device)
        # Create target from clean input
        clean_target = make_shift_targets(clean_input, shift_size).to(example_input.device)
        # Add noise to input
        noise_level = 0.1  # Adjust as needed
        # Add complex or real noise depending on input type
        if torch.is_complex(clean_input):
            noise = noise_level * torch.complex(torch.randn_like(clean_input.real), torch.randn_like(clean_input.imag))
        else:
            noise = noise_level * torch.randn_like(clean_input)
        noisy_input = clean_input + noise
        
        # Get model output on noisy input
        with torch.no_grad():
            noisy_output = model(noisy_input)
        
        # Convert to numpy
        noisy_input_np = noisy_input[0].cpu().numpy()
        noisy_output_np = noisy_output[0].cpu().numpy()
        clean_target_np = clean_target[0].cpu().numpy()
        
        noisy_io_fig_path = os.path.join(run_dir, "example_io_noisy.png")
        plot_noisy_example_io(noisy_input_np, noisy_output_np, clean_target_np, 
                             save_path=noisy_io_fig_path, title=fig_title)

    # Save JSON logs (including complex lambdas and c values converted to arrays)
    serializable_logs = {
        "steps": logs.steps,
        "losses": logs.losses,
        "lambdas": [[complex(z).real, complex(z).imag] for ckpt in logs.lambdas for z in ckpt],
        "c_values": [[complex(z).real, complex(z).imag] for ckpt in logs.c_values for z in ckpt]
    }
    # reshape back to (checkpoints, n, 2)
    C = len(logs.lambdas)
    Nn = len(logs.lambdas[0])
    serializable_logs["lambdas"] = np.array(serializable_logs["lambdas"], dtype=float).reshape(C, Nn, 2).tolist()
    serializable_logs["c_values"] = np.array(serializable_logs["c_values"], dtype=float).reshape(C, Nn, 2).tolist()

    # Include all hyperparameters
    all_meta = {
        "hyperparams": hyperparams,
        "init_radii": list(map(float, init_radii)),
        "init_angles": list(map(float, init_angles)),
        "target_frequencies": list(map(float, target_frequencies)),
    }
    with open(os.path.join(run_dir, "training_logs.json"), "w") as f:
        json.dump({**all_meta, "logs": serializable_logs}, f, indent=2)

    return run_dir

# %%
# One-call experiment runner

def run_experiment(
    n: int = N,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    shift_size: int = SHIFT_SIZE,
    num_steps: int = NUM_STEPS,
    optimizer_name: str = OPTIMIZER,
    lr_adam: float = LR_ADAM,
    lr_sgd: float = LR_SGD,
    weight_decay: float = WEIGHT_DECAY,
    target_frequencies: List[float] = TARGET_FREQUENCIES,
    init_radii: List[float] = INIT_RADII,
    init_angles: List[float] = INIT_ANGLES,
    base_path: str = BASE_PATH,
    run_name_prefix: str = RUN_NAME_PREFIX,
    device: str = DEVICE,
    learn_c: bool = True,  # whether to learn C (only if c_is_one is False)
    bc_real: bool = BC_REAL,  # whether B and C are real-valued
    b_is_one: bool = B_IS_ONE,  # whether B is fixed to 1
    c_is_one: bool = C_IS_ONE,  # whether C is fixed to 1
    take_real: bool = TAKE_REAL,  # whether to take real part of output
    noise_level: float = NOISE_LEVEL,  # std of noise to add to inputs during training
    loss_threshold: float = 1e-5,  # train until loss < this
    max_steps: int = 50000,  # maximum steps (safety limit)
) -> str:
    assert len(init_radii) == n and len(init_angles) == n, "init lists must match n"

    set_seed(SEED)

    model = DiagonalSSM(init_radii=init_radii, init_angles=init_angles, 
                       learn_c=learn_c, bc_real=bc_real, b_is_one=b_is_one, 
                       c_is_one=c_is_one, take_real=take_real).to(device)

    logs = train_shift_task(
        model=model,
        freqs=target_frequencies,
        seq_len=seq_len,
        batch_size=batch_size,
        shift_size=shift_size,
        num_steps=num_steps,
        optimizer_name=optimizer_name,
        lr_adam=lr_adam,
        lr_sgd=lr_sgd,
        weight_decay=weight_decay,
        log_every=LOG_EVERY,
        device=device,
        loss_threshold=loss_threshold,
        max_steps=max_steps,
        take_real=take_real,
        noise_level=noise_level,
    )

    # Generate an example input/target for visualization
    example_input = sample_frequency_mixture(target_frequencies, seq_len, batch_size, take_real=take_real).to(device)
    example_target = make_shift_targets(example_input, shift_size).to(device)

    # Bundle hyperparameters to save
    hps = dict(
        SEED=SEED, DEVICE=device,
        N=n, SEQ_LEN=seq_len, BATCH_SIZE=batch_size, SHIFT_SIZE=shift_size,
        NUM_STEPS=num_steps, OPTIMIZER=optimizer_name, LR_ADAM=lr_adam, LR_SGD=lr_sgd,
        WEIGHT_DECAY=weight_decay, LOG_EVERY=LOG_EVERY,
        LOSS_THRESHOLD=loss_threshold, MAX_STEPS=max_steps,
        LEARN_C=learn_c, BC_REAL=bc_real, B_IS_ONE=b_is_one, C_IS_ONE=c_is_one, TAKE_REAL=take_real,
        NOISE_LEVEL=noise_level,
    )

    run_dir = save_result(
        BASE_PATH=base_path,
        init_radii=init_radii,
        init_angles=init_angles,
        target_frequencies=target_frequencies,
        logs=logs,
        hyperparams=hps,
        run_name_prefix=run_name_prefix,
        model=model,
        example_input=example_input,
        example_target=example_target,
        shift_size=shift_size,
        seq_len=seq_len,
        batch_size=batch_size,
        take_real=take_real,
    )

    return run_dir

# %%
# Example run - edit these parameters easily:
loss_threshold = 1e-2
# Initial eigenvalue configuration
my_init_radii = [0.99]           # Radii in (0, 1) - one per eigenvalue
my_init_angles = [0.001]        # Angles in (-pi, pi] radians - one per eigenvalue

# Input frequency configuration (normalized: 0-1 where 1 = π radians)
my_input_freqs = [0.2]          # e.g., [0.1, 0.5, 1.0] → [0.1π, 0.5π, π] radians

# Run the experiment with the above settings
result_dir = run_experiment(
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
    noise_level=0,
)
print(f"Results saved to: {result_dir}")

# %%

# Initial eigenvalue configuration
my_init_angles = [np.pi+0.001]  # Angles in (-pi, pi] radians - one per eigenvalue

# Run the experiment with the above settings
result_dir = run_experiment(
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")

# %%

# Initial eigenvalue configuration
my_init_angles = [np.pi/2]  # Angles in (-pi, pi] radians - one per eigenvalue

# Run the experiment with the above settings
result_dir = run_experiment(
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")

# %%
my_init_radii = [0.9,0.9]           # Radii in (0, 1) - one per eigenvalue
my_init_angles = [0.001,np.pi-0.1*np.pi]  # Angles in (-pi, pi] radians - one per eigenvalue

# Run the experiment with the above settings
result_dir = run_experiment(
    n = len(my_init_radii),
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")
# %%
my_init_radii = [0.9,0.9]           # Radii in (0, 1) - one per eigenvalue
my_init_angles = [0.001,0.05]  # Angles in (-pi, pi] radians - one per eigenvalue

# Run the experiment with the above settings
result_dir = run_experiment(
    n = len(my_init_radii),
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")
# %%
# %%
# Example run - edit these parameters easily:
n = 15
loss_threshold = 3e0
# Initial eigenvalue configuration
my_init_radii = [0.9]*n           # Radii in (0, 1) - one per eigenvalue
my_init_angles = [0.001*(i+1) for i in range(n)]        # Angles in (-pi, pi] radians - one per eigenvalue

# Input frequency configuration (normalized: 0-1 where 1 = π radians)
my_input_freqs = [0.2, 0.1, 0.05, 0.01]          # e.g., [0.1, 0.5, 1.0] → [0.1π, 0.5π, π] radians

# Run the experiment with the above settings
result_dir = run_experiment(
    n = n,
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")

# %%
# # Example run - edit these parameters easily:
# loss_threshold = 5e-3
# # Initial eigenvalue configuration
my_init_radii = [0.9]*n           # Radii in (0, 1) - one per eigenvalue
my_init_angles = [0.1*(i+1) for i in range(n)]        # Angles in (-pi, pi] radians - one per eigenvalue

# Input frequency configuration (normalized: 0-1 where 1 = π radians)
my_input_freqs = [0.2, 0.1, 0.05, 0.01]          # e.g., [0.1, 0.5, 1.0] → [0.1π, 0.5π, π] radians

# Run the experiment with the above settings
result_dir = run_experiment(
    n = n,
    init_radii=my_init_radii,
    init_angles=my_init_angles,
    target_frequencies=my_input_freqs,
    optimizer_name="adam",
    loss_threshold=loss_threshold,
)
print(f"Results saved to: {result_dir}")
# %%
