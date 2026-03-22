"""
TEP Clinical Report Generator.
Uses Anthropic API when key is set; falls back to template.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import TEPMetricsExtractor


# ─── Template fallback ────────────────────────────────────────────────────

def build_template_report(metrics: dict) -> str:
    peaks = metrics.get('peaks', {})
    alpha = metrics.get('alpha_power_ratio', 1.0)
    hint  = metrics.get('interpretation_hint', '')

    lines = [
        "CLINICAL TEP INTERPRETATION",
        "=" * 44,

        "",
        "Motor Cortex (M1) TMS-EEG Recording",
        "",
        "Peak Metrics (Cz electrode):",
    ]

    for name, data in peaks.items():
        status = []
        if data.get('delayed'):
            status.append('DELAYED')
        if data.get('reduced'):
            status.append('REDUCED')
        flag = '  [' + ', '.join(status) + ']' if status else ''
        lines.append(
            f"  {name:6s}  lat={data['latency_ms']:6.1f} ms  "
            f"amp={data['amplitude_uv']:+7.3f} µV{flag}"
        )

    lines += [
        "",
        f"Alpha power ratio (post/pre): {alpha:.3f}",
        "",
        "Interpretation:",
    ]

    # Build narrative
    narrative = []

    n45 = peaks.get('N45', {})
    n100 = peaks.get('N100', {})

    if n45:
        if n45.get('delayed'):
            narrative.append(
                f"The N45 component shows delayed latency ({n45['latency_ms']:.1f} ms "
                f"vs normative ~45 ms), suggesting reduced GABAergic inhibitory "
                f"activity in M1."
            )
        else:
            narrative.append(
                f"The N45 component is within normative range ({n45['latency_ms']:.1f} ms), "
                f"consistent with intact GABAergic inhibition in M1."
            )
        if n45.get('reduced'):
            narrative.append(
                f"The reduced N45 amplitude ({n45['amplitude_uv']:.3f} µV) indicates "
                f"diminished GABA-B receptor-mediated inhibition."
            )

    if n100:
        if n100.get('reduced'):
            narrative.append(
                f"The N100 amplitude is attenuated ({n100['amplitude_uv']:.3f} µV), "
                f"which may reflect changes in long-range cortical inhibitory networks."
            )

    if alpha > 1.2:
        narrative.append(
            f"Post-stimulus alpha power increase (ratio={alpha:.2f}) suggests "
            f"cortical idling or inhibitory rebound following the TMS pulse."
        )
    elif alpha < 0.8:
        narrative.append(
            f"Post-stimulus alpha suppression (ratio={alpha:.2f}) reflects "
            f"enhanced cortical excitability following M1 stimulation."
        )

    if not narrative:
        narrative.append(
            "All TEP components are within normative ranges, indicating "
            "normal cortical excitability and inhibitory function in M1."
        )

    lines += narrative
    if hint and hint != "TEP components within normative range.":
        lines += ["", f"Clinical flags: {hint}"]

    best = metrics.get('best_model')
    f1   = metrics.get('f1_score')
    if best:
        lines += ["", f"Artifact model: {best}" + (f" (F1={f1:.3f})" if f1 else "")]

    return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────

def compute_tep_metrics(evoked, registry=None) -> dict:
    """Compute full TEP metrics using TEPMetricsExtractor."""
    extractor = TEPMetricsExtractor()
    return extractor.generate_metrics_report(evoked, registry=registry)


def generate_report(evoked, registry=None,
                    path: str = 'outputs/tep_report.txt') -> str:
    """
    Generate clinical TEP report.
    Tries Anthropic API first; falls back to template.
    Saves to path and prints to terminal.
    Returns the output path.
    """
    import numpy as np

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    metrics = compute_tep_metrics(evoked, registry=registry)

    # ── Try Anthropic API ─────────────────────────────────────
    report_text = None
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            # Summarise metrics for the prompt
            peaks_summary = {
                name: {k: v for k, v in d.items() if k in ('latency_ms', 'amplitude_uv')}
                for name, d in metrics.get('peaks', {}).items()
            }
            prompt = (
                f"TMS-EEG data from M1 stimulation.\n"
                f"Peak metrics (Cz): {peaks_summary}\n"
                f"Alpha power ratio (post/pre): {metrics.get('alpha_power_ratio', 'N/A')}\n"
                f"Artifact model: {metrics.get('best_model', 'N/A')} "
                f"(F1={metrics.get('f1_score', 'N/A')})\n"
                f"Please write a 4-sentence clinical interpretation."
            )

            msg = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=300,
                system=(
                    'You are a neuroscientist interpreting TMS-EEG results '
                    'from M1 stimulation. Write a 4-sentence clinical '
                    'interpretation. Be specific about peak latencies and what '
                    'they imply about cortical excitability.'
                ),
                messages=[{'role': 'user', 'content': prompt}]
            )
            ai_text = msg.content[0].text
            report_text = (
                "CLINICAL TEP INTERPRETATION (AI-generated)\n"
                "=" * 44 + "\n\n"
                + ai_text + "\n\n"
                "--- Numeric Metrics ---\n"
                + build_template_report(metrics).split("Interpretation:")[0]
            )
            print("[report] Anthropic API used for interpretation.")
        except Exception as e:
            print(f"[report] Anthropic API failed ({e}), using template.")

    if report_text is None:
        report_text = build_template_report(metrics)

    # ── Save ──────────────────────────────────────────────────
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + "-" * 50)
    print(report_text)
    print("-" * 50 + "\n")
    print(f"[report] Saved to {path}")
    return path
