"""
TMS-EEG Data Augmentation Pipeline (TEPAugmenter)

This module implements a comprehensive data augmentation pipeline for TMS-EEG
artifact detection. It provides 6 augmentation methods specifically designed
for TMS-evoked potential (TEP) data.

Author: TMS-EEG Signal Processing Pipeline
Date: 2024
"""

import numpy as np
from scipy import signal
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import warnings

# Constants for TMS-EEG data
SFREQ = 1000  # Sampling frequency in Hz
TMIN = -0.2   # Start time relative to TMS pulse (seconds)
TMAX = 0.5    # End time relative to TMS pulse (seconds)
N_CH = 19     # Number of EEG channels (10-20 system)
N_TIMES = 701 # Number of time samples (TMAX - TMIN) * SFREQ + 1

# Artifact window: 5ms to 50ms post-pulse
# With TMIN=-0.2, the pulse is at sample 200
# 5ms post-pulse = sample 205, 50ms post-pulse = sample 250
ARTIFACT_START_SAMPLE = int((-TMIN + 0.005) * SFREQ)  # 205
ARTIFACT_END_SAMPLE = int((-TMIN + 0.050) * SFREQ)    # 250


def generate_synthetic_data(
    n_trials: int = 300,
    artifact_ratio: float = 0.35,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic TMS-EEG data for testing the augmentation pipeline.
    
    This function creates realistic-looking TMS-EEG trials with:
    - Baseline noise (pink noise characteristic of EEG)
    - TMS-evoked potentials (TEPs) in clean trials
    - Muscle artifacts in artifact trials
    
    Parameters
    ----------
    n_trials : int
        Number of trials to generate.
    artifact_ratio : float
        Proportion of trials containing artifacts (label=0).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray
        Synthetic EEG data of shape (n_trials, n_channels, n_times).
    y : np.ndarray
        Binary labels (0=artifact, 1=clean).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_artifacts = int(n_trials * artifact_ratio)
    n_clean = n_trials - n_artifacts
    
    X = np.zeros((n_trials, N_CH, N_TIMES), dtype=np.float32)
    y = np.zeros(n_trials, dtype=np.int32)
    
    # Generate pink noise for all trials (1/f characteristic of EEG)
    for trial_idx in range(n_trials):
        for ch in range(N_CH):
            # Pink noise generation using inverse FFT
            white_noise = np.random.randn(N_TIMES)
            # Create 1/f filter in frequency domain
            freqs = np.fft.rfftfreq(N_TIMES)
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            # Apply filter
            fft_noise = np.fft.rfft(white_noise)
            pink_noise = np.fft.irfft(fft_noise * pink_filter, n=N_TIMES)
            # Normalize to microvolts (typical EEG amplitude)
            pink_noise = pink_noise / (np.std(pink_noise) + 1e-8) * 2.0
            X[trial_idx, ch] = pink_noise.astype(np.float32)
    
    # Add TEP (TMS-evoked potential) to clean trials
    time = np.linspace(TMIN, TMAX, N_TIMES)
    
    # TEP components: N45, P60, N100, P180
    tep_components = [
        ('N45', 0.045, 0.015, -2.5),   # Negative peak at 45ms
        ('P60', 0.060, 0.020, 3.0),    # Positive peak at 60ms
        ('N100', 0.100, 0.030, -4.0),  # Negative peak at 100ms
        ('P180', 0.180, 0.050, 2.5),   # Positive peak at 180ms
    ]
    
    clean_indices = list(range(n_artifacts, n_trials))
    for trial_idx in clean_indices:
        y[trial_idx] = 1
        # Vary TEP amplitude across channels and trials
        for ch in range(N_CH):
            ch_scale = 0.8 + 0.4 * np.random.rand()  # Channel-specific scaling
            trial_scale = 0.9 + 0.2 * np.random.rand()  # Trial variability
            
            for name, peak_time, sigma, amplitude in tep_components:
                # Gaussian-shaped component
                component = amplitude * ch_scale * trial_scale * np.exp(
                    -0.5 * ((time - peak_time) / sigma) ** 2
                )
                X[trial_idx, ch] += component.astype(np.float32)
    
    # Add muscle artifacts to artifact trials
    artifact_indices = list(range(n_artifacts))
    for trial_idx in artifact_indices:
        y[trial_idx] = 0
        
        # Muscle artifact: high-frequency burst 5-50ms post-pulse
        artifact_duration = ARTIFACT_END_SAMPLE - ARTIFACT_START_SAMPLE
        artifact_signal = np.zeros(N_TIMES)
        
        # Create muscle artifact with decaying envelope
        t_artifact = np.linspace(0, 1, artifact_duration)
        envelope = np.exp(-3 * t_artifact)  # Exponential decay
        
        # High-frequency oscillation (20-100 Hz typical for muscle)
        freq = 40 + 60 * np.random.rand()  # Random frequency 40-100 Hz
        carrier = np.sin(2 * np.pi * freq * t_artifact / 1000)
        
        # Amplitude varies by trial
        artifact_amp = 15 + 25 * np.random.rand()  # 15-40 µV
        artifact_signal[ARTIFACT_START_SAMPLE:ARTIFACT_END_SAMPLE] = (
            artifact_amp * envelope * carrier
        )
        
        # Apply to subset of channels (muscle artifact is spatially localized)
        n_affected_channels = max(1, int(0.4 * N_CH))  # 40% of channels
        affected_channels = np.random.choice(
            N_CH, n_affected_channels, replace=False
        )
        
        for ch in affected_channels:
            ch_artifact = artifact_signal * (0.7 + 0.6 * np.random.rand())
            X[trial_idx, ch] += ch_artifact.astype(np.float32)
    
    # Shuffle the trials
    shuffle_idx = np.random.permutation(n_trials)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


class TEPAugmenter:
    """
    TMS-Evoked Potential Data Augmenter.
    
    Implements 6 augmentation methods for TMS-EEG data:
    1. time_jitter - Temporal shift augmentation
    2. amplitude_scale - Channel-wise amplitude scaling
    3. channel_noise - SNR-calibrated noise injection
    4. muscle_artifact_inject - TMS muscle artifact simulation
    5. baseline_shift - DC offset augmentation
    6. trial_mixing - Mixup augmentation for same-class trials
    
    Parameters
    ----------
    sfreq : int
        Sampling frequency in Hz. Default: 1000.
    tmin : float
        Start time relative to TMS pulse in seconds. Default: -0.2.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        sfreq: int = SFREQ,
        tmin: float = TMIN,
        random_state: Optional[int] = None
    ):
        self.sfreq = sfreq
        self.tmin = tmin
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        
        # Compute artifact window indices
        self.artifact_start = int((-tmin + 0.005) * sfreq)
        self.artifact_end = int((-tmin + 0.050) * sfreq)
    
    def _reset_rng(self):
        """Reset random number generator to initial state."""
        self._rng = np.random.RandomState(self.random_state)
    
    def time_jitter(
        self,
        X: np.ndarray,
        max_ms: float = 8.0,
        sfreq: int = 1000
    ) -> np.ndarray:
        """
        Apply temporal jitter by shifting each trial randomly.
        
        Each trial is shifted by a random amount within ±max_ms using
        circular shifting (np.roll). This simulates slight timing
        variability in TMS pulse delivery and neural response latency.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        max_ms : float
            Maximum shift in milliseconds. Default: 8.0.
        sfreq : int
            Sampling frequency in Hz. Default: 1000.
        
        Returns
        -------
        X_jittered : np.ndarray
            Time-jittered data with same shape as input.
        """
        X_jittered = X.copy()
        max_samples = int(max_ms * sfreq / 1000)
        
        for i in range(X.shape[0]):
            # Random shift between -max_samples and +max_samples
            shift = self._rng.randint(-max_samples, max_samples + 1)
            # Apply same shift to all channels in a trial
            X_jittered[i] = np.roll(X[i], shift, axis=1)
        
        return X_jittered
    
    def amplitude_scale(
        self,
        X: np.ndarray,
        low: float = 0.75,
        high: float = 1.25
    ) -> np.ndarray:
        """
        Apply per-channel random amplitude scaling.
        
        Each channel in each trial is multiplied by a random scalar
        drawn uniformly from [low, high]. This simulates variations
        in electrode impedance and amplifier gain.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        low : float
            Lower bound of scaling factor. Default: 0.75.
        high : float
            Upper bound of scaling factor. Default: 1.25.
        
        Returns
        -------
        X_scaled : np.ndarray
            Amplitude-scaled data with same shape as input.
        """
        X_scaled = X.copy().astype(np.float32)
        n_trials, n_channels, n_times = X.shape
        
        # Generate random scaling factors for each trial and channel
        scales = self._rng.uniform(low, high, size=(n_trials, n_channels)).astype(np.float32)
        
        # Apply scaling (broadcasting over time dimension)
        X_scaled = X_scaled * scales[:, :, np.newaxis]
        
        return X_scaled
    
    def channel_noise(
        self,
        X: np.ndarray,
        snr_db: float = 18.0
    ) -> np.ndarray:
        """
        Add Gaussian noise calibrated to a target SNR.
        
        Noise is added independently to each channel with power set
        to achieve the specified Signal-to-Noise Ratio in dB.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        snr_db : float
            Target SNR in decibels. Default: 18.0.
        
        Returns
        -------
        X_noisy : np.ndarray
            Noise-augmented data with same shape as input.
        """
        X_noisy = X.copy()
        n_trials, n_channels, n_times = X.shape
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        for i in range(n_trials):
            for ch in range(n_channels):
                # Compute signal power
                signal_power = np.mean(X[i, ch] ** 2)
                
                # Compute noise power to achieve target SNR
                noise_power = signal_power / snr_linear
                noise_std = np.sqrt(noise_power)
                
                # Generate and add noise
                noise = self._rng.randn(n_times) * noise_std
                X_noisy[i, ch] += noise.astype(np.float32)
        
        return X_noisy
    
    def muscle_artifact_inject(
        self,
        X: np.ndarray,
        y: np.ndarray,
        p: float = 0.20,
        scale: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject TMS muscle artifact into a subset of trials.
        
        Muscle artifacts are characterized by high-frequency oscillations
        in the 5-50ms window post-TMS pulse. This method injects such
        artifacts into p% of trials on random 40% of channels.
        
        Important: Trials with injected artifacts are labeled as 0 (artifact).
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        y : np.ndarray
            Input labels of shape (n_trials,).
        p : float
            Proportion of trials to inject artifact into. Default: 0.20.
        scale : float
            Artifact amplitude scaling factor. Default: 1.5.
        
        Returns
        -------
        X_artifact : np.ndarray
            Data with injected muscle artifacts.
        y_artifact : np.ndarray
            Updated labels (injected trials are now artifact=0).
        """
        X_artifact = X.copy()
        y_artifact = y.copy()
        
        n_trials = X.shape[0]
        n_to_inject = int(n_trials * p)
        n_channels = X.shape[1]
        
        # Select random trials for artifact injection
        inject_indices = self._rng.choice(
            n_trials, n_to_inject, replace=False
        )
        
        # Artifact parameters
        artifact_duration = self.artifact_end - self.artifact_start
        t_artifact = np.linspace(0, 1, artifact_duration)
        
        for idx in inject_indices:
            # Number of channels to affect (40%)
            n_affected = max(1, int(0.4 * n_channels))
            affected_channels = self._rng.choice(
                n_channels, n_affected, replace=False
            )
            
            for ch in affected_channels:
                # Create muscle artifact with decaying envelope
                envelope = np.exp(-3 * t_artifact)
                
                # High-frequency oscillation (muscle artifact characteristic)
                freq = 40 + 60 * self._rng.rand()
                carrier = np.sin(2 * np.pi * freq * t_artifact / 1000)
                
                # Artifact amplitude based on existing signal level
                base_amp = np.std(X_artifact[idx, ch]) * scale
                artifact_amp = base_amp * (0.5 + self._rng.rand())
                
                artifact = artifact_amp * envelope * carrier
                
                # Inject artifact
                X_artifact[idx, ch, self.artifact_start:self.artifact_end] += (
                    artifact.astype(np.float32)
                )
            
            # Label this trial as artifact (0)
            y_artifact[idx] = 0
        
        return X_artifact, y_artifact
    
    def baseline_shift(
        self,
        X: np.ndarray,
        max_uv: float = 0.5
    ) -> np.ndarray:
        """
        Apply random DC offset (baseline shift) to each channel.
        
        This simulates slow drifts in EEG signals caused by
        electrode polarization, skin potentials, etc.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        max_uv : float
            Maximum baseline shift in microvolts. Default: 0.5.
        
        Returns
        -------
        X_shifted : np.ndarray
            Data with baseline shifts applied.
        """
        X_shifted = X.copy().astype(np.float32)
        n_trials, n_channels = X.shape[:2]
        
        # Generate random shifts for each trial and channel
        shifts = self._rng.uniform(
            -max_uv, max_uv, size=(n_trials, n_channels)
        ).astype(np.float32)
        
        # Apply shifts (broadcasting over time)
        X_shifted = X_shifted + shifts[:, :, np.newaxis]
        
        return X_shifted
    
    def trial_mixing(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation between same-class trials.
        
        Linear interpolation between pairs of trials from the same class:
        X_new = alpha * X_i + (1 - alpha) * X_j
        
        This generates new synthetic trials while preserving label integrity.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        y : np.ndarray
            Input labels of shape (n_trials,).
        alpha : float
            Mixing coefficient. Default: 0.2.
            If alpha=0.2, new trial = 0.2*X_i + 0.8*X_j.
        
        Returns
        -------
        X_mixed : np.ndarray
            Mixed trials with shape (n_mixed, n_channels, n_times).
        y_mixed : np.ndarray
            Labels for mixed trials (same as original class).
        """
        # Separate trials by class
        clean_idx = np.where(y == 1)[0]
        artifact_idx = np.where(y == 0)[0]
        
        mixed_trials = []
        mixed_labels = []
        
        # Mix clean trials
        n_clean = len(clean_idx)
        if n_clean >= 2:
            n_pairs = n_clean // 2
            self._rng.shuffle(clean_idx)
            
            for i in range(n_pairs):
                idx1, idx2 = clean_idx[2*i], clean_idx[2*i + 1]
                # Random mixing coefficient from Beta distribution
                lam = self._rng.beta(alpha * 10, alpha * 10)
                lam = np.clip(lam, 0.1, 0.9)  # Ensure meaningful mixing
                
                mixed_trial = lam * X[idx1] + (1 - lam) * X[idx2]
                mixed_trials.append(mixed_trial)
                mixed_labels.append(1)  # Clean label
        
        # Mix artifact trials
        n_artifact = len(artifact_idx)
        if n_artifact >= 2:
            n_pairs = n_artifact // 2
            self._rng.shuffle(artifact_idx)
            
            for i in range(n_pairs):
                idx1, idx2 = artifact_idx[2*i], artifact_idx[2*i + 1]
                lam = self._rng.beta(alpha * 10, alpha * 10)
                lam = np.clip(lam, 0.1, 0.9)
                
                mixed_trial = lam * X[idx1] + (1 - lam) * X[idx2]
                mixed_trials.append(mixed_trial)
                mixed_labels.append(0)  # Artifact label
        
        if len(mixed_trials) == 0:
            return np.array([]).reshape(0, X.shape[1], X.shape[2]), np.array([])
        
        X_mixed = np.stack(mixed_trials, axis=0).astype(np.float32)
        y_mixed = np.array(mixed_labels, dtype=np.int32)
        
        return X_mixed, y_mixed
    
    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factor: int = 3,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentation methods to produce factor× more data.
        
        This method:
        1. Applies each augmentation method separately
        2. Samples from augmented pool to reach exactly factor× size
        3. Balances classes
        4. Shuffles the final dataset
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_trials, n_channels, n_times).
        y : np.ndarray
            Input labels of shape (n_trials,).
        factor : int
            Multiplication factor for dataset size. Default: 3.
        seed : int
            Random seed for reproducibility. Default: 42.
        
        Returns
        -------
        X_aug : np.ndarray
            Augmented dataset with exactly factor× samples.
        y_aug : np.ndarray
            Corresponding labels.
        """
        self.random_state = seed
        self._rng = np.random.RandomState(seed)
        
        n_original = X.shape[0]
        target_total = n_original * factor
        
        # Collect augmented samples by class
        augmented_by_class = {0: [], 1: []}
        
        # Keep original data
        for label in [0, 1]:
            label_idx = np.where(y == label)[0]
            augmented_by_class[label].append(X[label_idx].copy())
        
        # 1. Time jitter - add to each class separately
        X_jitter = self.time_jitter(X, max_ms=8, sfreq=self.sfreq)
        for label in [0, 1]:
            label_idx = np.where(y == label)[0]
            augmented_by_class[label].append(X_jitter[label_idx])
        
        # 2. Amplitude scale
        X_scaled = self.amplitude_scale(X, low=0.75, high=1.25)
        for label in [0, 1]:
            label_idx = np.where(y == label)[0]
            augmented_by_class[label].append(X_scaled[label_idx])
        
        # 3. Channel noise
        X_noisy = self.channel_noise(X, snr_db=18)
        for label in [0, 1]:
            label_idx = np.where(y == label)[0]
            augmented_by_class[label].append(X_noisy[label_idx])
        
        # 4. Muscle artifact injection (converts some clean to artifact)
        X_artifact, y_artifact = self.muscle_artifact_inject(
            X.copy(), y.copy(), p=0.20, scale=1.5
        )
        for label in [0, 1]:
            label_idx = np.where(y_artifact == label)[0]
            augmented_by_class[label].append(X_artifact[label_idx])
        
        # 5. Baseline shift
        X_baseline = self.baseline_shift(X, max_uv=0.5)
        for label in [0, 1]:
            label_idx = np.where(y == label)[0]
            augmented_by_class[label].append(X_baseline[label_idx])
        
        # 6. Trial mixing (generates new trials within same class)
        X_mixed, y_mixed = self.trial_mixing(X, y, alpha=0.2)
        if X_mixed.shape[0] > 0:
            for label in [0, 1]:
                label_idx = np.where(y_mixed == label)[0]
                if len(label_idx) > 0:
                    augmented_by_class[label].append(X_mixed[label_idx])
        
        # Concatenate all augmented samples by class
        class_pools = {}
        for label in [0, 1]:
            class_pools[label] = np.concatenate(augmented_by_class[label], axis=0)
        
        # Calculate target samples per class for balanced output
        samples_per_class = target_total // 2
        
        # Sample exactly samples_per_class from each class
        final_X = []
        final_y = []
        
        for label in [0, 1]:
            pool = class_pools[label]
            n_available = pool.shape[0]
            
            if n_available >= samples_per_class:
                # Randomly sample without replacement
                selected_idx = self._rng.choice(
                    n_available, samples_per_class, replace=False
                )
            else:
                # Sample with replacement if not enough
                selected_idx = self._rng.choice(
                    n_available, samples_per_class, replace=True
                )
            
            final_X.append(pool[selected_idx])
            final_y.append(np.full(samples_per_class, label, dtype=np.int32))
        
        # Combine and shuffle
        X_combined = np.concatenate(final_X, axis=0).astype(np.float32)
        y_combined = np.concatenate(final_y, axis=0)
        
        # Shuffle
        shuffle_idx = self._rng.permutation(X_combined.shape[0])
        X_aug = X_combined[shuffle_idx]
        y_aug = y_combined[shuffle_idx]
        
        return X_aug, y_aug


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets.
    
    Uses stratified splitting to maintain class balance across splits.
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_trials, n_channels, n_times).
    y : np.ndarray
        Labels of shape (n_trials,).
    val_frac : float
        Fraction of data for validation. Default: 0.2.
    test_frac : float
        Fraction of data for testing. Default: 0.1.
    seed : int
        Random seed. Default: 42.
    
    Returns
    -------
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray
        Validation data and labels.
    X_test, y_test : np.ndarray
        Test data and labels.
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    
    # Second split: separate validation from remaining
    val_frac_adjusted = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_adjusted, 
        random_state=seed, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for handling imbalanced datasets.
    
    Uses sklearn's balanced class weighting scheme:
    weight = n_samples / (n_classes * n_samples_for_class)
    
    Parameters
    ----------
    y : np.ndarray
        Labels of shape (n_trials,).
    
    Returns
    -------
    class_weights : dict
        Dictionary mapping class labels to their weights.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    class_weights = dict(zip(classes, weights))
    
    return class_weights


def print_class_distribution(y: np.ndarray, name: str = "Dataset"):
    """Print the class distribution of a dataset."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"\n{name} Class Distribution:")
    print(f"  Total trials: {total}")
    for label, count in zip(unique, counts):
        pct = 100 * count / total
        label_name = "Clean" if label == 1 else "Artifact"
        print(f"  {label_name} (label={label}): {count} ({pct:.1f}%)")


def main():
    """
    Main function to test the TMS-EEG augmentation pipeline.
    
    Generates 300 synthetic trials, augments to 900, and saves to file.
    """
    print("=" * 60)
    print("TMS-EEG Data Augmentation Pipeline")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic TMS-EEG data...")
    X, y = generate_synthetic_data(n_trials=300, artifact_ratio=0.35, seed=42)
    print(f"    Generated {X.shape[0]} trials")
    print(f"    Data shape: {X.shape}")
    print(f"    Label shape: {y.shape}")
    
    # Print original distribution
    print_class_distribution(y, "Original Dataset")
    
    # Create augmenter
    augmenter = TEPAugmenter(sfreq=SFREQ, tmin=TMIN, random_state=42)
    
    # Test individual augmentation methods
    print("\n[2] Testing individual augmentation methods...")
    
    # Time jitter test
    X_jitter = augmenter.time_jitter(X[:5], max_ms=8, sfreq=SFREQ)
    print(f"    time_jitter: shape={X_jitter.shape}, dtype={X_jitter.dtype}")
    
    # Amplitude scale test
    X_scaled = augmenter.amplitude_scale(X[:5], low=0.75, high=1.25)
    print(f"    amplitude_scale: shape={X_scaled.shape}, dtype={X_scaled.dtype}")
    
    # Channel noise test
    X_noisy = augmenter.channel_noise(X[:5], snr_db=18)
    print(f"    channel_noise: shape={X_noisy.shape}, dtype={X_noisy.dtype}")
    
    # Muscle artifact inject test
    X_artifact, y_artifact = augmenter.muscle_artifact_inject(
        X[:10], y[:10], p=0.30, scale=1.5
    )
    print(f"    muscle_artifact_inject: X={X_artifact.shape}, y={y_artifact.shape}")
    
    # Baseline shift test
    X_baseline = augmenter.baseline_shift(X[:5], max_uv=0.5)
    print(f"    baseline_shift: shape={X_baseline.shape}, dtype={X_baseline.dtype}")
    
    # Trial mixing test
    X_mixed, y_mixed = augmenter.trial_mixing(X, y, alpha=0.2)
    print(f"    trial_mixing: X={X_mixed.shape}, y={y_mixed.shape}")
    
    # Augment dataset
    print("\n[3] Augmenting dataset (factor=3)...")
    X_aug, y_aug = augmenter.augment_dataset(X, y, factor=3, seed=42)
    
    print(f"    Augmented shape: {X_aug.shape}")
    print(f"    Augmented labels: {y_aug.shape}")
    
    # Print augmented distribution
    print_class_distribution(y_aug, "Augmented Dataset")
    
    # Compute class weights
    print("\n[4] Computing class weights...")
    class_weights = compute_class_weights(y_aug)
    print(f"    Class weights: {class_weights}")
    
    # Split dataset
    print("\n[5] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X_aug, y_aug, val_frac=0.2, test_frac=0.1, seed=42
    )
    
    print(f"    Train: {X_train.shape[0]} trials")
    print(f"    Val:   {X_val.shape[0]} trials")
    print(f"    Test:  {X_test.shape[0]} trials")
    
    print_class_distribution(y_train, "Training Set")
    print_class_distribution(y_val, "Validation Set")
    print_class_distribution(y_test, "Test Set")
    
    # Save augmented dataset
    output_path = "/home/z/my-project/data/augmented.npz"
    print(f"\n[6] Saving augmented dataset to {output_path}...")
    np.savez(
        output_path,
        X_aug=X_aug,
        y_aug=y_aug,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        class_weights=np.array([class_weights[0], class_weights[1]])
    )
    print("    Saved successfully!")
    
    # Verification
    print("\n[7] Verifying saved data...")
    loaded = np.load(output_path)
    print(f"    Loaded X_aug shape: {loaded['X_aug'].shape}")
    print(f"    Loaded y_aug shape: {loaded['y_aug'].shape}")
    print(f"    Class weights: {loaded['class_weights']}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return X_aug, y_aug


if __name__ == "__main__":
    main()
