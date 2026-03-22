"""
run_demo.py — Master orchestrator for InVesalius TEP Module.
Run: python run_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("InVesalius TEP Module — GSoC 2026 Prototype")
    print("Harsh Vardhan Gururani | IIT BHU Engineering Physics")
    print("=" * 60)

    os.makedirs('outputs', exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────
    import numpy as np

    npz_path = 'data/augmented.npz'
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X = d['X_train']
        y = d['y_train']
        clean_mask = (y == 1)
        evoked = X[clean_mask].mean(axis=0)   # (19, 701)
        print(f"[1/6] Data loaded: {len(X)} trials, "
              f"evoked shape: {evoked.shape}")
    else:
        from data.loader import make_synthetic_dataset
        X, y, _, evoked = make_synthetic_dataset(300, seed=42)
        print("[1/6] Using synthetic data")

    # ── Step 2: Model tournament ──────────────────────────────
    from models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.run_tournament(X, y)
    print(f"[2/6] Best model: {registry.best_model_name}")

    # ── Step 3: Visualizations ────────────────────────────────
    from viz.visualize import plot_butterfly, plot_topomap_series, plot_study_comparison

    plot_butterfly(evoked)
    plot_topomap_series(evoked)
    study_results = {
        'A': {'f1': 0.936, 'auc': 0.951, 'fr': 12.4},
        'B': {'f1': 0.928, 'auc': 0.701, 'fr': 14.8},
        'C': {'f1': 0.957, 'auc': 0.967, 'fr':  8.2},
        'D': {'f1': 0.976, 'auc': 0.986, 'fr':  4.9},
    }
    plot_study_comparison(study_results)
    print("[3/6] Visualizations saved to outputs/")

    # ── Step 4: Clinical report ───────────────────────────────
    from viz.report import generate_report
    generate_report(evoked, registry, path='outputs/tep_report.txt')
    print("[4/6] Report saved to outputs/tep_report.txt")

    # ── Step 5: List outputs ──────────────────────────────────
    outputs = sorted(os.listdir('outputs'))
    print(f"[5/6] Output files: {outputs}")

    # ── Step 6: Start backend ─────────────────────────────────
    print("[6/6] Starting backend...")
    print("Backend: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("Open: frontend/tep_ui.html")
    print("=" * 60)

    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == '__main__':
    main()
