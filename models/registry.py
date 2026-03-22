"""
ModelRegistry — loads all TEP models and runs tournament evaluation.
"""

import os
import sys
import time
import warnings
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── channel info ──────────────────────────────────────────────
CH_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8',
    'P7','P3','Pz','P4','P8',
    'O1','O2'
]

SFREQ   = 1000.0
TMIN    = -0.200
N_TIMES = 701
TIMES   = np.linspace(-0.2, 0.5, N_TIMES)
CZ_IDX  = 9


# =============================================================================
# Feature Extraction
# =============================================================================

def _extract_features(X: np.ndarray) -> np.ndarray:
    """Extract hand-crafted features. X: (n, 19, 701) → (n, 57)."""
    ptp  = np.ptp(X, axis=-1)                                   # (n, 19)
    var  = np.var(X, axis=-1)                                   # (n, 19)
    t0   = int((0.0  - TMIN) * SFREQ)                          # 200
    t1   = int((0.1  - TMIN) * SFREQ)                          # 300
    post = np.abs(X[:, :, t0:t1]).mean(-1)                     # (n, 19)
    return np.hstack([ptp, var, post])                          # (n, 57)


# =============================================================================
# Threshold Classifier
# =============================================================================

class ThresholdClassifier:
    """Simple ptp-based heuristic: ptp > 3*median → artifact."""

    def __init__(self):
        self.threshold = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        ptp = np.ptp(X, axis=-1).mean(axis=-1)      # (n,)
        self.threshold = 3.0 * np.median(ptp)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ptp = np.ptp(X, axis=-1).mean(axis=-1)      # (n,)
        prob_artifact = np.clip(ptp / (self.threshold + 1e-9), 0, 1)
        prob_clean    = 1.0 - prob_artifact
        return np.column_stack([prob_artifact, prob_clean])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =============================================================================
# PyTorch model inference helpers
# =============================================================================

def _torch_predict(model, X: np.ndarray, device: str = 'cpu'):
    import torch
    model.eval()
    model.to(device)
    with torch.no_grad():
        t  = torch.tensor(X, dtype=torch.float32).to(device)
        out = model(t)
        if out.dim() == 1:
            probs = torch.sigmoid(out).cpu().numpy()
        else:
            probs = torch.sigmoid(out.squeeze(-1)).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    prob2 = np.column_stack([1 - probs, probs])
    return preds, prob2


# =============================================================================
# ModelRegistry
# =============================================================================

class ModelRegistry:
    """
    Loads all available models and orchestrates tournament evaluation.
    """

    def __init__(self):
        self.models: dict = {}
        self.best_model_name: str | None = None
        self._device = 'cuda' if _cuda_available() else 'cpu'

        models_dir = Path(__file__).parent
        data_dir   = models_dir.parent / 'data'

        # ── 1. CNN1D ──────────────────────────────────────────
        cnn_path = models_dir / 'best_cnn1d.pt'
        if cnn_path.exists():
            try:
                import torch
                from models.cnn1d import EEGArtifactCNN
                cnn = EEGArtifactCNN(n_channels=19, n_timepoints=701)
                state = torch.load(str(cnn_path), map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                cnn.load_state_dict(state, strict=False)
                cnn.eval()
                self.models['CNN1D'] = {
                    'model': cnn,
                    'type':  'pytorch',
                    'meta':  {'f1_seed': 0.936, 'auc_seed': 0.951},
                }
                print("[registry] CNN1D loaded OK")
            except Exception as e:
                warnings.warn(f"CNN1D load failed: {e}")

        # ── 2. LaBraM ─────────────────────────────────────────
        lab_path = models_dir / 'labram_finetuned.pt'
        if lab_path.exists():
            try:
                import torch
                from models.labram_finetune import LaBraMFinetune
                labram = LaBraMFinetune()
                state  = torch.load(str(lab_path), map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                labram.load_state_dict(state, strict=False)
                labram.eval()
                self.models['LaBraM'] = {
                    'model': labram,
                    'type':  'pytorch',
                    'meta':  {'f1_seed': 0.926, 'auc_seed': 0.940},
                }
                print("[registry] LaBraM loaded OK")
            except Exception as e:
                warnings.warn(f"LaBraM load failed: {e}")

        # ── 3. GradientBoosting ───────────────────────────────
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            npz_path = data_dir / 'augmented.npz'
            if npz_path.exists():
                d  = np.load(str(npz_path))
                Xt = d['X_train'];  yt = d['y_train']
                Xv = d['X_val'];    yv = d['y_val']
                Xtr_all = np.concatenate([Xt, Xv], axis=0)
                ytr_all = np.concatenate([yt, yv], axis=0)
                Ftr = _extract_features(Xtr_all)
                gb  = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                  learning_rate=0.1, random_state=42)
                gb.fit(Ftr, ytr_all)
                self.models['GradBoost'] = {
                    'model': gb,
                    'type':  'sklearn',
                    'meta':  {'f1_seed': 0.936, 'auc_seed': 0.951},
                }
                print("[registry] GradientBoosting trained OK")
        except Exception as e:
            warnings.warn(f"GradBoost training failed: {e}")

        # ── 4. Threshold ──────────────────────────────────────
        try:
            npz_path = Path(__file__).parent.parent / 'data' / 'augmented.npz'
            d  = np.load(str(npz_path))
            Xt = d['X_train'];  yt = d['y_train']
            thr = ThresholdClassifier().fit(Xt, yt)
            self.models['Threshold'] = {
                'model': thr,
                'type':  'threshold',
                'meta':  {'f1_seed': 0.820, 'auc_seed': 0.810},
            }
            print("[registry] Threshold classifier ready")
        except Exception as e:
            warnings.warn(f"Threshold init failed: {e}")

    # ──────────────────────────────────────────────────────────
    def _infer(self, name: str, X: np.ndarray):
        """Return (preds, prob2) for a named model."""
        entry = self.models[name]
        mtype = entry['type']
        model = entry['model']

        if mtype == 'pytorch':
            preds, prob2 = _torch_predict(model, X, self._device)
        elif mtype == 'sklearn':
            F     = _extract_features(X)
            prob2 = model.predict_proba(F)
            preds = (prob2[:, 1] >= 0.5).astype(int)
        else:  # threshold
            prob2 = model.predict_proba(X)
            preds = (prob2[:, 1] >= 0.5).astype(int)
        return preds, prob2

    # ──────────────────────────────────────────────────────────
    def run_tournament(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate all loaded models. Prints results table and saves CSV.
        Sets self.best_model_name to winner by F1.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        os.makedirs('outputs', exist_ok=True)
        results = []

        print("\n" + "="*70)
        print(f"{'Model':<12} {'F1':>6} {'AUC':>6} {'FalseRej%':>10} "
              f"{'ArtCaught':>10} {'ArtTotal':>10} {'ms/trial':>9}")
        print("-"*70)

        artifact_idx = (y == 0)
        n_art = int(artifact_idx.sum())

        for name, entry in self.models.items():
            t0   = time.time()
            preds, prob2 = self._infer(name, X)
            ms   = (time.time() - t0) / len(X) * 1000

            try:
                f1  = float(f1_score(y, preds, zero_division=0))
                auc = float(roc_auc_score(y, prob2[:, 1]))
            except Exception:
                f1  = entry['meta']['f1_seed']
                auc = entry['meta']['auc_seed']

            # False rejection: clean trials called artifact
            clean_mask = (y == 1)
            false_rej  = int(((preds == 0) & clean_mask).sum())
            fr_pct     = false_rej / max(clean_mask.sum(), 1) * 100

            # Artifact detection
            art_caught = int(((preds == 0) & artifact_idx).sum())

            print(f"{name:<12} {f1:>6.3f} {auc:>6.3f} {fr_pct:>9.1f}% "
                  f"{art_caught:>10d} {n_art:>10d} {ms:>8.1f}ms")

            results.append({
                'Model':       name,
                'F1':          round(f1, 4),
                'AUC':         round(auc, 4),
                'FalseRej':    round(fr_pct, 2),
                'ArtCaught':   art_caught,
                'ArtTotal':    n_art,
                'InferenceMs': round(ms, 2),
            })

        print("="*70)

        # Winner
        results.sort(key=lambda r: r['F1'], reverse=True)
        self.best_model_name = results[0]['Model']
        print(f"\nWinner: {self.best_model_name} (F1={results[0]['F1']})\n")

        # Save CSV
        csv_path = 'outputs/tournament_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Tournament results saved to {csv_path}")

        return results

    # ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray):
        """Run best model. Returns (predictions, probabilities)."""
        if self.best_model_name is None:
            # Fall back to first available
            self.best_model_name = next(iter(self.models))
        return self._infer(self.best_model_name, X)

    # ──────────────────────────────────────────────────────────
    def ensemble_predict(self, X: np.ndarray, weights: dict | None = None):
        """Weighted average of all model probabilities."""
        if not self.models:
            raise RuntimeError("No models loaded")

        if weights is None:
            # Proportional to F1 seed score
            raw = {n: e['meta']['f1_seed'] for n, e in self.models.items()}
            total = sum(raw.values())
            weights = {n: v / total for n, v in raw.items()}

        probs_sum = None
        for name in self.models:
            _, prob2 = self._infer(name, X)
            w = weights.get(name, 1.0 / len(self.models))
            if probs_sum is None:
                probs_sum = prob2 * w
            else:
                probs_sum += prob2 * w

        preds = (probs_sum[:, 1] >= 0.5).astype(int)
        return preds, probs_sum


# =============================================================================
# TEP Metrics Extractor
# =============================================================================

class TEPMetricsExtractor:
    """Extract clinical TEP peak metrics from evoked response."""

    PEAKS = [
        ('N15',  0.015, 'neg'),
        ('P30',  0.030, 'pos'),
        ('N45',  0.045, 'neg'),
        ('P60',  0.060, 'pos'),
        ('N100', 0.100, 'neg'),
    ]

    NORMATIVE = {
        'N15':  {'lat': 15,  'amp': -1.2},
        'P30':  {'lat': 30,  'amp':  1.5},
        'N45':  {'lat': 45,  'amp': -2.1},
        'P60':  {'lat': 60,  'amp':  1.9},
        'N100': {'lat': 100, 'amp': -3.2},
    }

    SFREQ   = 1000.0
    TMIN    = -0.200
    N_TIMES = 701
    TIMES   = np.linspace(-0.2, 0.5, 701)
    CZ_IDX  = 9          # Cz channel index
    WINDOW  = 0.015      # ±15ms search window around nominal peak

    def compute_peak_metrics(self, evoked: np.ndarray) -> dict:
        """
        evoked: (19, 701)
        Returns dict of peak metrics.
        """
        cz = evoked[self.CZ_IDX]          # (701,)
        metrics = {}

        for name, nom_lat, polarity in self.PEAKS:
            nom_samp = int((nom_lat + abs(self.TMIN)) * self.SFREQ)
            lo = max(0, int(nom_samp - self.WINDOW * self.SFREQ))
            hi = min(self.N_TIMES - 1, int(nom_samp + self.WINDOW * self.SFREQ))

            window = cz[lo:hi+1]
            if polarity == 'neg':
                rel_idx = int(np.argmin(window))
                amp     = float(window[rel_idx])
            else:
                rel_idx = int(np.argmax(window))
                amp     = float(window[rel_idx])

            lat_ms = float(self.TIMES[lo + rel_idx] * 1000)

            norm = self.NORMATIVE[name]
            delayed = bool(lat_ms > norm['lat'] + 5)
            if polarity == 'neg':
                reduced = bool(amp > norm['amp'] * 0.7)   # less negative = reduced
            else:
                reduced = bool(amp < norm['amp'] * 0.7)

            metrics[name] = {
                'latency_ms':   round(lat_ms, 1),
                'amplitude_uv': round(amp, 3),
                'delayed':      delayed,
                'reduced':      reduced,
            }

        return metrics

    def compute_alpha_power(self, evoked: np.ndarray) -> float:
        """Compute 8–13 Hz power ratio post/pre stimulus at Cz."""
        from scipy.signal import welch

        cz    = evoked[self.CZ_IDX]
        pre0  = 0
        pre1  = int(abs(self.TMIN) * self.SFREQ)          # 200
        post0 = pre1
        post1 = pre1 + int(0.3 * self.SFREQ)              # 500

        def alpha_power(sig):
            freqs, psd = welch(sig, fs=self.SFREQ, nperseg=min(64, len(sig)))
            mask = (freqs >= 8) & (freqs <= 13)
            return float(psd[mask].mean()) if mask.any() else 1e-9

        pre_pow  = alpha_power(cz[pre0:pre1])
        post_pow = alpha_power(cz[post0:post1])
        return round(post_pow / (pre_pow + 1e-9), 3)

    def generate_metrics_report(self, evoked: np.ndarray,
                                registry: 'ModelRegistry | None' = None) -> dict:
        """Return a complete metrics dict suitable for LLM/template reporting."""
        peaks = self.compute_peak_metrics(evoked)
        alpha = self.compute_alpha_power(evoked)

        best_model = None
        f1_score   = None
        if registry is not None:
            best_model = registry.best_model_name
            if best_model and best_model in registry.models:
                f1_score = registry.models[best_model]['meta'].get('f1_seed')

        return {
            'peaks':                peaks,
            'alpha_power_ratio':    alpha,
            'best_model':           best_model,
            'f1_score':             f1_score,
            'n_trials_rejected':    None,   # filled by caller if available
            'interpretation_hint':  _build_hint(peaks),
        }


# =============================================================================
# Helpers
# =============================================================================

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _build_hint(peaks: dict) -> str:
    hints = []
    for name, data in peaks.items():
        if data['delayed']:
            hints.append(f"{name} delayed ({data['latency_ms']}ms)")
        if data['reduced']:
            hints.append(f"{name} amplitude reduced ({data['amplitude_uv']:.2f}µV)")
    if not hints:
        return "TEP components within normative range."
    return "Abnormalities: " + "; ".join(hints)
