# %%
# Experiment: Sweep initial angles for diagonal complex SSM
# Based on single_exp.py but runs multiple experiments with varying initial angles
# Creates animated GIF showing impulse response evolution
# Author: ChatGPT for Yuval Ran-Milo (@yuv.milo)

# %%
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from PIL import Image

# ---------------------------
# Hyperparameters
# ---------------------------

SEED = 3
DEVICE = "cpu"

# Experiment management
CLEAR_EXP = True  # Set to True to clear all previous experiments on startup
BASE_PATH = "./ssm_exp_many"  # results root

# Model architecture flags
BC_REAL = False
B_IS_ONE = True
C_IS_ONE = False
TAKE_REAL = False  # Use complex throughout

# SSM size and sequence/batch config
N = 1  # dimension of the SSM (1D for this sweep)
SEQ_LEN = 64
BATCH_SIZE = 256
SHIFT_SIZE = 0

# Training config
NUM_STEPS = 500
OPTIMIZER = "adam"
LR_ADAM = 5e-3
LR_SGD = 1e-2
WEIGHT_DECAY = 0.0
NOISE_LEVEL = 0.0
LOG_EVERY = 10

# Fixed parameters for the sweep
FIXED_RADIUS = 0.9
FIXED_INPUT_FREQ = 0.2  # normalized frequency (0 to 1)

# Angle sweep configuration
NUM_ANGLES = 50  # number of angles to test
ANGLE_MIN = 0.0  # start angle (radians, will be slightly offset)
ANGLE_MAX = 2 * np.pi  # end angle (radians, will be slightly offset)

# GIF configuration
GIF_DURATION = 20  # seconds
GIF_FPS = NUM_ANGLES / GIF_DURATION

# Loss threshold
LOSS_THRESHOLD = 1e-2
MAX_STEPS = 50000

# ---------------------------
# End hyperparameters section
# ---------------------------

# %%
import math

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
    """Clear all previous experiment results."""
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Cleared all experiments in {base_path}")
    else:
        print(f"No experiments found at {base_path}")

# Clear experiments if requested
if CLEAR_EXP:
    clear_experiments()

os.makedirs(BASE_PATH, exist_ok=True)

# %%
# Utility functions from single_exp.py

def stable_radius_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """LRU-like stable exponential parameterization."""
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
                            take_real: bool = False) -> torch.Tensor:
    """Create batch of frequency mixtures."""
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # (1, T)
    Fm = torch.tensor(freqs, dtype=torch.float32).view(-1, 1) * math.pi  # (M, 1)
    
    if take_real:
        basis_bank = torch.cos(Fm * t)
        coeffs = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        mix = coeffs @ basis_bank
    else:
        basis_bank = torch.exp(1j * Fm * t)
        coeffs_real = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        coeffs_imag = torch.randn(batch_size, len(freqs), dtype=torch.float32)
        coeffs = torch.complex(coeffs_real, coeffs_imag)
        mix = coeffs @ basis_bank

    norms = torch.linalg.norm(mix, ord=2, dim=1, keepdim=True) + 1e-12
    return mix

def make_shift_targets(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift sequences to the right by `shift` with zero-padding."""
    b, T = x.shape
    y = torch.zeros_like(x)
    if shift <= 0:
        return x
    y[:, shift:] = x[:, :-shift]
    return y

# %%
# Model

class DiagonalSSM(nn.Module):
    """Complex diagonal State-Space / linear RNN."""
    def __init__(self, init_radii: List[float], init_angles: List[float], 
                 learn_c: bool=True, bc_real: bool=False, b_is_one: bool=True, 
                 c_is_one: bool=False, take_real: bool=False):
        super().__init__()
        assert len(init_radii) == len(init_angles), "radii and angles lengths must match"
        n = len(init_radii)
        self.n = n
        self.bc_real = bc_real
        self.b_is_one = b_is_one
        self.c_is_one = c_is_one
        self.take_real = take_real

        r = torch.tensor(init_radii, dtype=torch.float32)
        r = torch.clamp(r, 1e-6, 1-1e-6)
        rho0 = torch.log(torch.expm1(-torch.log(r)))
        theta0 = torch.tensor(init_angles, dtype=torch.float32)

        self.rho   = nn.Parameter(rho0)
        self.theta = nn.Parameter(theta0)

        dtype = torch.float32 if bc_real else torch.complex64
        self.learn_c = learn_c
        
        if b_is_one:
            self.b = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=False)
        else:
            self.b = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=True)
        
        if c_is_one:
            self.c = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=False)
        else:
            self.c = nn.Parameter(torch.ones(n, dtype=dtype), requires_grad=learn_c)

    def eigenvalues(self) -> torch.Tensor:
        """Return λ (complex) as vector shape (n,)"""
        r = stable_radius_from_rho(self.rho)
        lam = complex_diag_from_rt(r.to(self.theta.device), self.theta)
        return lam.to(torch.complex64)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : (batch, T) real or complex
        returns y : (batch, T) real if take_real=True, complex otherwise
        """
        batch, T = u.shape
        lam = self.eigenvalues()
        x = torch.zeros(batch, self.n, dtype=torch.complex64, device=u.device)
        
        output_dtype = torch.float32 if self.take_real else torch.complex64
        y = torch.zeros(batch, T, dtype=output_dtype, device=u.device)

        b = self.b.to(x.device)
        c = self.c.to(x.device)
        if self.bc_real:
            b = torch.complex(b, torch.zeros_like(b))
            c = torch.complex(c, torch.zeros_like(c))

        for t in range(T):
            u_t = u[:, t].unsqueeze(-1)
            if not torch.is_complex(u_t):
                u_t = torch.complex(u_t, torch.zeros_like(u_t))
            
            x = lam * x + b * u_t
            y_t = torch.matmul(x, torch.conj(c))
            
            if self.take_real:
                y[:, t] = torch.real(y_t)
            else:
                y[:, t] = y_t

        return y

# %%
# Training

def complex_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss for both real and complex tensors."""
    if torch.is_complex(output) or torch.is_complex(target):
        diff = output - target
        return torch.mean(torch.real(diff * torch.conj(diff)))
    else:
        return F.mse_loss(output, target)

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
    device: str = "cpu",
    loss_threshold: float = 1e-5,
    max_steps: int = 50000,
    take_real: bool = False,
    noise_level: float = 0.0,
    verbose: bool = False,
) -> float:
    """Train and return final loss."""
    model.to(device)
    model.train()

    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.0, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    pbar = tqdm(range(1, max_steps + 1), desc="Training", unit="step", disable=not verbose)
    
    final_loss = float('nan')
    for step in pbar:
        u = sample_frequency_mixture(freqs, seq_len, batch_size, take_real=take_real).to(device)
        
        if noise_level > 0:
            if torch.is_complex(u):
                noise = noise_level * torch.complex(
                    torch.randn_like(u.real, device=device),
                    torch.randn_like(u.imag, device=device)
                )
            else:
                noise = noise_level * torch.randn_like(u, device=device)
            u_noisy = u + noise
        else:
            u_noisy = u
        
        y_target = make_shift_targets(u_noisy, shift_size).to(device)
        y_hat = model(u_noisy)
        loss = complex_mse_loss(y_hat, y_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        current_loss = float(loss.item())
        final_loss = current_loss
        
        if verbose:
            pbar.set_postfix({"loss": f"{current_loss:.2e}", "threshold": f"{loss_threshold:.2e}"})

        if current_loss < loss_threshold:
            if verbose:
                pbar.set_postfix({"loss": f"{current_loss:.2e}", "status": "converged!"})
                pbar.close()
                print(f"\nConverged at step {step} with loss {current_loss:.2e}")
            break
    else:
        if verbose:
            pbar.close()
            print(f"\nReached max steps ({max_steps}) with final loss {current_loss:.2e}")

    return final_loss

# %%
# Directory management

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_sweep_run_dir(
    base_path: str,
    run_name_prefix: str,
    n: int,
    fixed_radii: List[float],
    target_freqs: List[float],
    num_angles: int,
    loss_threshold: float,
    optimizer_name: str,
    shift_size: int = 0,
) -> str:
    """
    Make a human-readable directory name with key experiment parameters.
    Similar to make_run_dir in single_exp.py but for angle sweep experiments.
    """
    def fmt_list(xs, fmt="{:.3f}", k=5):
        xs = list(xs)
        tail = "" if len(xs) <= k else f"_plus{len(xs)-k}"
        return "_".join(fmt.format(v) for v in xs[:k]) + tail

    ts = time.strftime("%Y%m%d-%H%M%S")
    dir_name = (
        f"{run_name_prefix}"
        f"__n{n}"
        f"__shift{shift_size}"
        f"__freqs_{fmt_list(target_freqs)}"
        f"__radii_{fmt_list(fixed_radii)}"
        f"__nangles{num_angles}"
        f"__loss{loss_threshold:.0e}"
        f"__{optimizer_name}"
        f"__{ts}"
    )
    run_dir = os.path.join(base_path, dir_name)
    _ensure_dir(run_dir)
    return run_dir

# %%
# Impulse response computation

def compute_impulse_response(model: DiagonalSSM, seq_len: int, device: str = "cpu") -> np.ndarray:
    """
    Compute the impulse response of the model.
    Returns: complex numpy array of shape (seq_len,)
    """
    model.eval()
    
    # Create impulse input: [1, 0, 0, 0, ...]
    impulse = torch.zeros(1, seq_len, dtype=torch.complex64, device=device)
    impulse[0, 0] = 1.0
    
    # Get model response
    with torch.no_grad():
        response = model(impulse)  # (1, T)
    
    return response[0].cpu().numpy()

# %%
# Plotting function for combined visualization

def plot_impulse_and_eigenvalue(
    response_np: np.ndarray,
    init_eigenvalue: complex,
    final_eigenvalue: complex,
    init_angle: float,
    input_freq: float = None,
    save_path: str = "",
):
    """
    Create a combined plot with:
    - Time domain (real and imaginary parts)
    - Frequency domain (DTFT magnitude and phase)
    - Unit circle with eigenvalue position
    
    Args:
        input_freq: Primary input frequency (for single-frequency case). Can be None.
    """
    seq_len = len(response_np)
    
    # Compute DTFT
    response_fft = np.fft.fft(response_np)
    freqs = np.fft.fftfreq(seq_len, d=1.0)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Time domain - Real part
    ax1 = fig.add_subplot(gs[0, 0])
    time_steps = np.arange(seq_len)
    ax1.stem(time_steps, np.real(response_np), linefmt='b-', markerfmt='bo', basefmt='k-', label='Real part')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Response (real)", fontsize=11)
    ax1.set_title("Time Domain - Real", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc='best')
    
    # Time domain - Imaginary part
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.stem(time_steps, np.imag(response_np), linefmt='r-', markerfmt='ro', basefmt='k-', label='Imag part')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("Response (imag)", fontsize=11)
    ax2.set_title("Time Domain - Imaginary", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc='best')
    
    # Frequency domain - Magnitude (full spectrum, both positive and negative)
    ax3 = fig.add_subplot(gs[0, 1])
    # Sort frequencies for proper display
    sorted_idx = np.argsort(freqs)
    sorted_freqs = freqs[sorted_idx]
    sorted_mag = np.abs(response_fft[sorted_idx])
    ax3.plot(sorted_freqs, sorted_mag, 'g-', linewidth=2, label='|H(f)|')
    
    # Mark the input frequency if provided
    if input_freq is not None:
        input_freq_normalized = input_freq  # already in [0, 1]
        ax3.axvline(x=input_freq_normalized, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Input freq ({input_freq_normalized:.3f})')
        ax3.axvline(x=-input_freq_normalized, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel("Normalized Frequency", fontsize=11)
    ax3.set_ylabel("Magnitude", fontsize=11)
    ax3.set_title("DTFT - Magnitude", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc='best')
    ax3.set_xlim([-0.5, 0.5])
    
    # Frequency domain - Phase
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_phase = np.angle(response_fft[sorted_idx])
    ax4.plot(sorted_freqs, sorted_phase, 'm-', linewidth=2, label='∠H(f)')
    if input_freq is not None:
        ax4.axvline(x=input_freq_normalized, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Input freq ({input_freq_normalized:.3f})')
        ax4.axvline(x=-input_freq_normalized, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel("Normalized Frequency", fontsize=11)
    ax4.set_ylabel("Phase (radians)", fontsize=11)
    ax4.set_title("DTFT - Phase", fontsize=11, fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc='best')
    ax4.set_xlim([-0.5, 0.5])
    
    # Unit circle with eigenvalue
    ax5 = fig.add_subplot(gs[:, 2])
    
    # Draw unit circle
    thetas = np.linspace(0, 2*np.pi, 512)
    ax5.plot(np.cos(thetas), np.sin(thetas), 'k--', linewidth=1.5, alpha=0.4, label='Unit circle')
    
    # Plot initial eigenvalue (larger marker)
    ax5.plot(np.real(init_eigenvalue), np.imag(init_eigenvalue), 'o', 
            color='blue', markersize=15, markeredgecolor='black', markeredgewidth=2,
            label='Initial λ', zorder=5)
    
    # Plot final eigenvalue (star marker)
    ax5.plot(np.real(final_eigenvalue), np.imag(final_eigenvalue), '*', 
            color='red', markersize=20, markeredgecolor='black', markeredgewidth=2,
            label='Final λ', zorder=5)
    
    # Draw arrow from initial to final
    if np.abs(final_eigenvalue - init_eigenvalue) > 0.01:
        ax5.annotate('', xy=(np.real(final_eigenvalue), np.imag(final_eigenvalue)),
                    xytext=(np.real(init_eigenvalue), np.imag(init_eigenvalue)),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
    
    # Mark input frequency on unit circle if provided
    if input_freq is not None:
        input_angle = input_freq * np.pi
        ax5.plot(np.cos(input_angle), np.sin(input_angle), 's', 
                color='purple', markersize=12, markeredgecolor='black', markeredgewidth=2,
                label=f'Input freq ({input_freq:.2f}π)', zorder=5)
    
    ax5.set_xlabel("Real", fontsize=12)
    ax5.set_ylabel("Imaginary", fontsize=12)
    ax5.set_title("Unit Circle & Eigenvalue", fontsize=11, fontweight='bold')
    ax5.set_aspect("equal", adjustable="box")
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.legend(loc='best', fontsize=9)
    ax5.set_xlim([-1.2, 1.2])
    ax5.set_ylim([-1.2, 1.2])
    
    # Overall title
    freq_str = f"Input freq: {input_freq:.3f}π | " if input_freq is not None else ""
    title = (f"Init angle: {init_angle:.4f} rad ({np.degrees(init_angle):.2f}°) | "
            f"{freq_str}Radius: {np.abs(init_eigenvalue):.3f}")
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

# %%
# Main experiment loop

def run_angle_sweep_experiment(
    # Sweep configuration
    angles: List[float] = None,  # List of angles to sweep over
    num_angles: int = NUM_ANGLES,  # If angles not provided, generate this many
    angle_min: float = ANGLE_MIN,  # Min angle for linspace generation
    angle_max: float = ANGLE_MAX,  # Max angle for linspace generation
    
    # Fixed SSM parameters
    n: int = N,
    fixed_radii: List[float] = None,  # If None, uses [FIXED_RADIUS] * n
    
    # Input configuration
    target_frequencies: List[float] = None,  # If None, uses [FIXED_INPUT_FREQ]
    
    # Training configuration
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    shift_size: int = SHIFT_SIZE,
    num_steps: int = NUM_STEPS,
    optimizer_name: str = OPTIMIZER,
    lr_adam: float = LR_ADAM,
    lr_sgd: float = LR_SGD,
    weight_decay: float = WEIGHT_DECAY,
    loss_threshold: float = LOSS_THRESHOLD,
    max_steps: int = MAX_STEPS,
    noise_level: float = NOISE_LEVEL,
    
    # Model architecture
    learn_c: bool = True,
    bc_real: bool = BC_REAL,
    b_is_one: bool = B_IS_ONE,
    c_is_one: bool = C_IS_ONE,
    take_real: bool = TAKE_REAL,
    
    # Output configuration
    base_path: str = BASE_PATH,
    run_name_prefix: str = "angle_sweep",
    gif_duration: float = GIF_DURATION,
    device: str = DEVICE,
    seed: int = SEED,
) -> str:
    """
    Main experiment: sweep initial angles and create animated GIF.
    
    Args:
        angles: Explicit list of angles to test. If None, generates from num_angles/angle_min/angle_max
        num_angles: Number of angles to generate if angles not provided
        angle_min: Minimum angle for generation (radians)
        angle_max: Maximum angle for generation (radians)
        n: Dimension of SSM (number of eigenvalues)
        fixed_radii: List of radii (one per dimension). If None, uses [FIXED_RADIUS] * n
        target_frequencies: List of input frequencies. If None, uses [FIXED_INPUT_FREQ]
        loss_threshold: Training stops when loss < this threshold
        optimizer_name: "adam" or "sgd"
        run_name_prefix: Prefix for the run directory name
        ... (other standard training parameters)
        
    Returns:
        Path to the experiment directory containing all results (GIF, frames, logs)
    """
    # Set defaults
    if fixed_radii is None:
        fixed_radii = [FIXED_RADIUS] * n
    if target_frequencies is None:
        target_frequencies = [FIXED_INPUT_FREQ]
    
    assert len(fixed_radii) == n, f"fixed_radii must have length {n}"
    
    print(f"Starting angle sweep experiment...")
    print(f"SSM dimension (n): {n}")
    print(f"Fixed radii: {fixed_radii}")
    print(f"Input frequencies: {[f'{f}π' for f in target_frequencies]}")
    print(f"Loss threshold: {loss_threshold}")
    print(f"Optimizer: {optimizer_name}")
    
    # Generate angles if not provided
    if angles is None:
        # Exclude exact endpoints
        angles = np.linspace(angle_min, angle_max, num_angles + 2)[1:-1].tolist()
        print(f"Generated {len(angles)} angles from {angle_min:.4f} to {angle_max:.4f} rad")
    else:
        print(f"Using {len(angles)} provided angles")
    
    # Create run directory with descriptive name
    run_dir = make_sweep_run_dir(
        base_path=base_path,
        run_name_prefix=run_name_prefix,
        n=n,
        fixed_radii=fixed_radii,
        target_freqs=target_frequencies,
        num_angles=len(angles),
        loss_threshold=loss_threshold,
        optimizer_name=optimizer_name,
        shift_size=shift_size,
    )
    print(f"Output directory: {run_dir}")
    
    # Create subdirectory for frames
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_paths = []
    results_log = []
    
    for i, angle in enumerate(angles):
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{len(angles)}: Angle = {angle:.4f} rad ({np.degrees(angle):.2f}°)")
        print(f"{'='*70}")
        
        set_seed(seed)  # Reset seed for reproducibility
        
        # For sweep, we vary only the first angle
        init_angles = [angle] + [0.0] * (n - 1) if n > 1 else [angle]
        
        # Create model
        model = DiagonalSSM(
            init_radii=fixed_radii,
            init_angles=init_angles,
            learn_c=learn_c,
            bc_real=bc_real,
            b_is_one=b_is_one,
            c_is_one=c_is_one,
            take_real=take_real
        ).to(device)
        
        # Record initial eigenvalue (first one, which is the swept one)
        with torch.no_grad():
            init_eigenvalue = model.eigenvalues()[0].cpu().numpy()
        
        # Train model with verbose=True to show progress bar
        final_loss = train_shift_task(
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
            device=device,
            loss_threshold=loss_threshold,
            max_steps=max_steps,
            take_real=take_real,
            noise_level=noise_level,
            verbose=True,  # Enable progress bar for each training run
        )
        
        # Record final eigenvalue
        with torch.no_grad():
            final_eigenvalue = model.eigenvalues()[0].cpu().numpy()
        
        # Print summary
        print(f"Final loss: {final_loss:.4e}")
        print(f"Initial eigenvalue: {init_eigenvalue:.4f}")
        print(f"Final eigenvalue:   {final_eigenvalue:.4f}")
        print(f"Eigenvalue shift:   {np.abs(final_eigenvalue - init_eigenvalue):.4f}")
        
        # Compute impulse response
        response = compute_impulse_response(model, seq_len, device)
        
        # Create frame
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        print(f"Saving frame {i+1}/{len(angles)}...")
        plot_impulse_and_eigenvalue(
            response_np=response,
            init_eigenvalue=complex(init_eigenvalue),
            final_eigenvalue=complex(final_eigenvalue),
            init_angle=angle,
            input_freq=target_frequencies[0] if len(target_frequencies) == 1 else None,
            save_path=frame_path,
        )
        frame_paths.append(frame_path)
        
        # Log results
        results_log.append({
            'angle': float(angle),
            'init_eigenvalue_real': float(np.real(init_eigenvalue)),
            'init_eigenvalue_imag': float(np.imag(init_eigenvalue)),
            'final_eigenvalue_real': float(np.real(final_eigenvalue)),
            'final_eigenvalue_imag': float(np.imag(final_eigenvalue)),
            'final_loss': float(final_loss),
        })
    
    # Save results log
    with open(os.path.join(run_dir, "experiment_log.json"), 'w') as f:
        json.dump({
            'n': n,
            'fixed_radii': fixed_radii,
            'target_frequencies': target_frequencies,
            'num_angles': len(angles),
            'loss_threshold': loss_threshold,
            'optimizer': optimizer_name,
            'shift_size': shift_size,
            'results': results_log,
        }, f, indent=2)
    
    # Create GIF
    print(f"\n{'='*70}")
    print("Creating animated GIF...")
    print(f"{'='*70}")
    images = [Image.open(fp) for fp in tqdm(frame_paths, desc="Loading frames")]
    
    # Calculate duration per frame in milliseconds
    duration_per_frame = int((gif_duration * 1000) / len(angles))
    
    gif_path = os.path.join(run_dir, "angle_sweep_animation.gif")
    print("Saving GIF (this may take a moment)...")
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_per_frame,
        loop=0
    )
    
    # Print final summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"GIF saved to: {gif_path}")
    print(f"Total frames: {len(frame_paths)}")
    print(f"Duration: {gif_duration} seconds")
    print(f"FPS: {len(angles) / gif_duration:.2f}")
    print(f"All results saved to: {run_dir}")
    
    # Compute and show statistics
    final_losses = [r['final_loss'] for r in results_log]
    print(f"\nLoss statistics:")
    print(f"  Min loss: {min(final_losses):.4e}")
    print(f"  Max loss: {max(final_losses):.4e}")
    print(f"  Mean loss: {np.mean(final_losses):.4e}")
    print(f"  Converged (< {loss_threshold:.0e}): {sum(1 for l in final_losses if l < loss_threshold)}/{len(final_losses)}")
    print(f"{'='*70}\n")
    
    return run_dir

# %%
# Example: Custom angle range with explicit angles
custom_angles = np.linspace(0, 2*np.pi, 50)[1:-1]

for rad in [0.9, 0.95, 0.99, 0.999]:
    result_dir = run_angle_sweep_experiment(
        angles=custom_angles.tolist(),  # Explicit list of angles
        loss_threshold=1e-2,            # Tighter convergence
        n=1,                            # SSM dimension
        target_frequencies=[0.2],       # Input frequency
        optimizer_name="sgd",           # Use SGD instead of Adam
        fixed_radii=[rad],             # Fixed radius
    )
    print(f"\nResults directory: {result_dir}")
    print(f"GIF location: {result_dir}/angle_sweep_animation.gif")

# %%
