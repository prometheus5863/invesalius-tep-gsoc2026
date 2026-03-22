"""
TMS-EEG Data Loader — InVesalius TEP Module
Handles Rogasch figshare dataset (.mat format, both v5 and v7.3/HDF5)
Falls back to synthetic data if real data unavailable.
Author: Harsh Vardhan Gururani, IIT BHU — GSoC 2026
"""
import numpy as np
import os
import glob
from scipy.signal import butter, filtfilt

# ── CONSTANTS ────────────────────────────────────────────
SFREQ    = 1000.0
TMIN     = -0.200
TMAX     =  0.500
N_TIMES  = int((TMAX - TMIN) * SFREQ) + 1   # 701
TIMES    = np.linspace(TMIN, TMAX, N_TIMES)
N_CH     = 19

CH_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8',
            'P7','P3','Pz','P4','P8','O1','O2']

CH_POS = {
    'Fp1':(-0.30, 0.82), 'Fp2':( 0.30, 0.82),
    'F7': (-0.71, 0.42), 'F3': (-0.39, 0.48),
    'Fz': ( 0.00, 0.53), 'F4': ( 0.39, 0.48),
    'F8': ( 0.71, 0.42), 'T7': (-0.87, 0.00),
    'C3': (-0.49, 0.00), 'Cz': ( 0.00, 0.00),
    'C4': ( 0.49, 0.00), 'T8': ( 0.87, 0.00),
    'P7': (-0.71,-0.42), 'P3': (-0.39,-0.48),
    'Pz': ( 0.00,-0.53), 'P4': ( 0.39,-0.48),
    'P8': ( 0.71,-0.42), 'O1': (-0.30,-0.82),
    'O2': ( 0.30,-0.82),
}

MOTOR_CH = {'Cz', 'C3', 'C4', 'Fz', 'Pz'}

TEP_PEAKS = [
    {'name': 'N15',  't': 0.015, 'amp': -1.2, 'width': 0.008, 'col': '#ff453a'},
    {'name': 'P30',  't': 0.030, 'amp':  1.6, 'width': 0.010, 'col': '#ff9f0a'},
    {'name': 'N45',  't': 0.045, 'amp': -2.3, 'width': 0.012, 'col': '#bf5af2'},
    {'name': 'P60',  't': 0.060, 'amp':  1.9, 'width': 0.014, 'col': '#0a84ff'},
    {'name': 'N100', 't': 0.100, 'amp': -3.1, 'width': 0.020, 'col': '#30d158'},
]


# ── REAL DATA LOADER ─────────────────────────────────────

def load_rogasch_dataset(data_dir: str, max_subjects: int = 20):
    """
    Load the Rogasch TMS-EEG open dataset.
    Handles both MATLAB v5 (.mat via scipy) and v7.3 (.mat via h5py).
    
    Dataset: https://figshare.com/articles/dataset/TEPs-_SEPs/7440713
    
    Returns
    -------
    X : np.ndarray, shape (n_trials_total, 19, 701)
    y : np.ndarray, shape (n_trials_total,) — 1=clean, 0=artifact
    subjects : list of subject IDs
    evoked : np.ndarray, shape (19, 701) — grand average
    """
    mat_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.mat'),
                                  recursive=True))[:max_subjects]
    
    if not mat_files:
        print(f"  No .mat files found in {data_dir}")
        print("  Using synthetic fallback...")
        return make_synthetic_dataset(n_trials=200, seed=42)

    print(f"  Found {len(mat_files)} .mat files")
    
    # Try scipy first (v5 format), then h5py (v7.3/HDF5 format)
    all_X, all_y, subjects = [], [], []
    
    for fpath in mat_files:
        subj_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            result = _load_single_mat_scipy(fpath)
            if result is None:
                result = _load_single_mat_h5py(fpath)
        except Exception as e:
            print(f"  scipy failed for {subj_id}: {e}")
            try:
                result = _load_single_mat_h5py(fpath)
            except Exception as e2:
                print(f"  h5py also failed for {subj_id}: {e2}")
                continue
        
        if result is not None:
            X_subj, y_subj = result
            all_X.append(X_subj)
            all_y.append(y_subj)
            subjects.append(subj_id)
            print(f"  Loaded {subj_id}: {len(X_subj)} trials "
                  f"({sum(y_subj==0)} artifacts)")

    if not all_X:
        print("  All real data loads failed. Using synthetic.")
        return make_synthetic_dataset(n_trials=200, seed=42)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Preprocess
    X = _preprocess(X)
    evoked = X[y == 1].mean(axis=0)
    
    print(f"\n  Dataset summary:")
    print(f"  Total trials  : {len(X)}")
    print(f"  Clean trials  : {sum(y==1)}")
    print(f"  Artifact trials: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    print(f"  Subjects      : {len(subjects)}")
    
    return X, y, subjects, evoked


def _load_single_mat_scipy(fpath):
    """Load single .mat file using scipy (MATLAB v5 format)."""
    import scipy.io
    mat = scipy.io.loadmat(fpath, squeeze_me=True)
    keys = [k for k in mat.keys() if not k.startswith('_')]
    
    # Try common Rogasch field names
    data_arr = None
    for key in ['EEG', 'data', 'eeg', 'TEP']:
        if key not in mat:
            continue
        obj = mat[key]
        
        # Struct format: EEG.data is (n_ch, n_times, n_trials)
        if hasattr(obj, 'dtype') and obj.dtype.names:
            if 'data' in obj.dtype.names:
                data_arr = obj['data'].item()
                break
        # Direct array
        elif isinstance(obj, np.ndarray) and obj.ndim == 3:
            data_arr = obj
            break
    
    if data_arr is None:
        return None
    
    # Ensure shape is (n_trials, n_ch, n_times)
    if data_arr.ndim == 3:
        if data_arr.shape[0] < data_arr.shape[2]:
            # shape is (n_ch, n_times, n_trials) → transpose
            data_arr = data_arr.transpose(2, 0, 1)
        # Clip to our time window
        data_arr = data_arr[:, :N_CH, :N_TIMES]
    
    n_trials = data_arr.shape[0]
    # Heuristic artifact labels from amplitude
    ptp = np.ptp(data_arr, axis=-1).mean(axis=-1)
    thresh = 3.0 * np.median(ptp)
    y = (ptp < thresh).astype(int)
    
    return data_arr.astype(np.float32), y


def _load_single_mat_h5py(fpath):
    """Load single .mat file using h5py (MATLAB v7.3/HDF5 format)."""
    import h5py
    with h5py.File(fpath, 'r') as f:
        # Print structure on first call to help debugging
        print(f"  HDF5 keys in {os.path.basename(fpath)}: {list(f.keys())}")
        
        # Try to find the data array
        data_arr = None
        for key in ['EEG', 'data', 'eeg', 'TEP']:
            if key not in f:
                continue
            obj = f[key]
            if hasattr(obj, 'keys'):
                # Group (struct)
                if 'data' in obj:
                    data_arr = obj['data'][:]
                    break
            else:
                # Dataset
                data_arr = obj[:]
                break
        
        if data_arr is None:
            # Try first dataset found
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, 'shape') and len(obj.shape) == 3:
                    data_arr = obj[:]
                    break
        
        if data_arr is None:
            return None
        
        # HDF5 MATLAB stores arrays transposed
        data_arr = data_arr.T  # now (n_trials, n_ch, n_times) hopefully
        if data_arr.shape[1] > data_arr.shape[2]:
            data_arr = data_arr.transpose(0, 2, 1)
        
        data_arr = data_arr[:, :N_CH, :N_TIMES]
        ptp = np.ptp(data_arr, axis=-1).mean(axis=-1)
        thresh = 3.0 * np.median(ptp)
        y = (ptp < thresh).astype(int)
        
        return data_arr.astype(np.float32), y


def _preprocess(X):
    """Bandpass 1-100Hz + baseline correction per trial."""
    b, a = butter(4, [1.0/(SFREQ/2), 100.0/(SFREQ/2)], btype='band')
    bl_end = int((-0.010 - TMIN) * SFREQ)  # -10ms
    
    X_out = np.zeros_like(X)
    for i in range(len(X)):
        for ch in range(X.shape[1]):
            sig = filtfilt(b, a, X[i, ch])
            bl_mean = sig[:bl_end].mean()
            X_out[i, ch] = sig - bl_mean
    return X_out


# ── SYNTHETIC DATA GENERATOR ─────────────────────────────

def make_synthetic_dataset(n_trials: int = 300,
                            artifact_rate: float = 0.18,
                            noise_level: float = 1.8,
                            seed: int = 42):
    """
    Generate synthetic TMS-EEG data with realistic TEP structure.
    Used when real dataset not available or for augmentation.
    
    Returns same format as load_rogasch_dataset.
    """
    rng = np.random.RandomState(seed)
    b, a = butter(4, [1.0/(SFREQ/2), 100.0/(SFREQ/2)], btype='band')
    
    X, y = [], []
    art_start = int((0.005 - TMIN) * SFREQ)   # 205
    art_end   = int((0.050 - TMIN) * SFREQ)   # 250
    
    for trial in range(n_trials):
        data = np.zeros((N_CH, N_TIMES), dtype=np.float32)
        
        for ci, ch in enumerate(CH_NAMES):
            sw = (0.9 + rng.rand()*0.2) if ch in MOTOR_CH \
                 else (0.35 + rng.rand()*0.45)
            
            # TEP signal
            sig = sum(
                p['amp'] * sw * np.exp(-0.5*((TIMES - p['t'])/p['width'])**2)
                for p in TEP_PEAKS
            )
            # Pink-ish noise
            noise = filtfilt(b, a, rng.randn(N_TIMES)) * noise_level
            data[ci] = (sig + noise).astype(np.float32)
        
        # Artifact injection
        is_art = rng.rand() < artifact_rate
        if is_art:
            art_chs = rng.choice(N_CH, size=int(N_CH * 0.45), replace=False)
            data[art_chs, art_start:art_end] += \
                (rng.randn(len(art_chs), art_end - art_start) * 1.8).astype(np.float32)
            y.append(0)
        else:
            y.append(1)
        
        X.append(data)
    
    X = np.array(X)
    y = np.array(y)
    evoked = X[y == 1].mean(axis=0)
    
    print(f"  Synthetic dataset: {n_trials} trials, "
          f"{sum(y==0)} artifacts ({sum(y==0)/n_trials*100:.1f}%)")
    
    return X, y, [f'synth_{i:02d}' for i in range(5)], evoked


# ── INSPECT HELPER — paste output to Claude ─────────────

def inspect_mat_file(fpath: str):
    """
    Print the internal structure of a .mat file.
    Run this on your Rogasch .mat file and paste output to Claude.
    """
    print(f"\nInspecting: {os.path.basename(fpath)}")
    print(f"File size: {os.path.getsize(fpath)/1e6:.1f} MB")
    
    # Try scipy
    try:
        import scipy.io
        mat = scipy.io.loadmat(fpath, squeeze_me=True)
        keys = [k for k in mat.keys() if not k.startswith('_')]
        print(f"\nFormat: MATLAB v5 (scipy)")
        print(f"Top-level keys: {keys}")
        for k in keys:
            v = mat[k]
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            elif hasattr(v, 'dtype') and v.dtype.names:
                print(f"  {k}: struct with fields={v.dtype.names}")
                for field in v.dtype.names:
                    sub = v[field].item() if v.ndim > 0 else v[field]
                    if hasattr(sub, 'shape'):
                        print(f"    .{field}: shape={sub.shape} dtype={sub.dtype}")
        return
    except Exception as e:
        print(f"scipy failed: {e}")
    
    # Try h5py
    try:
        import h5py
        with h5py.File(fpath, 'r') as f:
            print(f"\nFormat: MATLAB v7.3 / HDF5 (h5py)")
            def show(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  {name}/  (group)")
            f.visititems(show)
    except Exception as e:
        print(f"h5py also failed: {e}")
        print("Install h5py: pip install h5py")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # python -m data.loader path/to/file.mat
        inspect_mat_file(sys.argv[1])
    else:
        # Test with synthetic data
        print("Testing synthetic data generation...")
        X, y, subjs, evoked = make_synthetic_dataset(n_trials=200, seed=42)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}, unique: {np.unique(y, return_counts=True)}")
        print(f"Evoked shape: {evoked.shape}")
        print(f"N45 amplitude at Cz: "
              f"{evoked[CH_NAMES.index('Cz'), int((0.045-TMIN)*SFREQ)]*1e6:.2f} µV")
        print("Synthetic data OK.")
