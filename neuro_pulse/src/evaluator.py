import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Evaluator Module.

Generates research paper figures and tables:
  - ROC curve (Figure 1)
  - SNR distribution histogram (Figure 2)
  - LaTeX results table
  - HTML report
"""

import argparse
import base64
import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.batch_analyzer import analyze_video


def load_dataset(real_dir: str, fake_dir: str) -> List[Dict]:
    """Load video paths with ground-truth labels."""
    dataset = []
    for ext in ("*.mp4", "*.mov", "*.avi"):
        for path in sorted(glob.glob(os.path.join(real_dir, ext))):
            dataset.append({"path": path, "ground_truth_label": "REAL"})
        for path in sorted(glob.glob(os.path.join(fake_dir, ext))):
            dataset.append({"path": path, "ground_truth_label": "FAKE"})
    return dataset


def run_evaluation(
    dataset: List[Dict],
    snr_threshold: float = 3.0,
) -> Tuple[Dict, List[Dict]]:
    """Run evaluation on the full dataset."""
    per_file_results = []
    tp = tn = fp = fn = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(dataset, desc="Evaluating")
    except ImportError:
        iterator = dataset

    for i, item in enumerate(iterator):
        result = analyze_video(item["path"], snr_threshold=snr_threshold)
        result["ground_truth"] = item["ground_truth_label"]
        per_file_results.append(result)

        verdict = result.get("verdict", "ERROR")
        truth = item["ground_truth_label"]

        if verdict == "ERROR":
            continue
        if verdict == "LIVE HUMAN" and truth == "REAL":
            tp += 1
        elif verdict == "SYNTHETIC" and truth == "FAKE":
            tn += 1
        elif verdict == "LIVE HUMAN" and truth == "FAKE":
            fp += 1
        elif verdict == "SYNTHETIC" and truth == "REAL":
            fn += 1

        if not hasattr(iterator, "set_description") and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # AUC
    labels = []
    scores = []
    for r in per_file_results:
        if r["ground_truth"] == "REAL":
            labels.append(1)
        elif r["ground_truth"] == "FAKE":
            labels.append(0)
        else:
            continue
        scores.append(r.get("snr_db", 0.0))

    auc = 0.0
    if len(labels) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.0

    metrics = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy, "tpr": tpr, "tnr": tnr,
        "fpr": fpr, "fnr": fnr, "auc": auc,
    }

    return metrics, per_file_results


def plot_roc_curve(per_file_results: List[Dict], save_path: str) -> None:
    """Generate ROC curve (paper Figure 1)."""
    labels = []
    scores = []
    for r in per_file_results:
        if r.get("verdict") == "ERROR":
            continue
        labels.append(1 if r["ground_truth"] == "REAL" else 0)
        scores.append(r.get("snr_db", 0.0))

    if len(labels) < 2:
        print("  WARNING: Not enough data for ROC curve")
        return

    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("  WARNING: scikit-learn not installed, skipping ROC curve")
        return

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr_arr, tpr_arr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, color="#1f77b4", linewidth=2,
            label=f"Neuro-Pulse (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")

    # Mark 3.0 dB operating point
    idx_3db = np.argmin(np.abs(thresholds - 3.0))
    ax.plot(fpr_arr[idx_3db], tpr_arr[idx_3db], "ro", markersize=10,
            label=f"3.0 dB threshold (FPR={fpr_arr[idx_3db]:.2f}, TPR={tpr_arr[idx_3db]:.2f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Neuro-Pulse Deepfake Detection", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ROC curve saved to {save_path}")


def plot_snr_distribution(per_file_results: List[Dict], save_path: str) -> None:
    """Generate SNR distribution histogram (paper Figure 2)."""
    real_snr = [r["snr_db"] for r in per_file_results
                if r["ground_truth"] == "REAL" and r.get("verdict") != "ERROR"]
    fake_snr = [r["snr_db"] for r in per_file_results
                if r["ground_truth"] == "FAKE" and r.get("verdict") != "ERROR"]

    if len(real_snr) == 0 and len(fake_snr) == 0:
        print("  WARNING: No data for SNR distribution")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(min(real_snr, default=0), min(fake_snr, default=0)) - 2,
        max(max(real_snr, default=10), max(fake_snr, default=10)) + 2,
        30,
    )

    if real_snr:
        ax.hist(real_snr, bins=bins, alpha=0.6, color="#2D7D4E", label="Real Faces",
                edgecolor="black", linewidth=0.5)
    if fake_snr:
        ax.hist(fake_snr, bins=bins, alpha=0.6, color="#CC2222", label="Deepfakes",
                edgecolor="black", linewidth=0.5)

    ax.axvline(3.0, color="red", linestyle="--", linewidth=2, label="Threshold (3.0 dB)")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SNR Distribution — Real vs Deepfake", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  SNR distribution saved to {save_path}")


def generate_latex_table(metrics: Dict) -> str:
    """Generate LaTeX table for the paper."""
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Neuro-Pulse Detection Performance}\n"
        "\\label{tab:results}\n"
        "\\begin{tabular}{|l|r|}\n"
        "\\hline\n"
        "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        "\\hline\n"
        f"Accuracy & {metrics['accuracy']:.2f} \\\\\n"
        f"True Positive Rate & {metrics['tpr']:.2f} \\\\\n"
        f"True Negative Rate & {metrics['tnr']:.2f} \\\\\n"
        f"False Positive Rate & {metrics['fpr']:.2f} \\\\\n"
        f"False Negative Rate & {metrics['fnr']:.2f} \\\\\n"
        f"AUC & {metrics.get('auc', 0.0):.2f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def generate_html_report(
    metrics: Dict,
    roc_path: str,
    dist_path: str,
) -> str:
    """Generate self-contained HTML report with embedded images."""
    def img_to_base64(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return ""

    roc_b64 = img_to_base64(roc_path)
    dist_b64 = img_to_base64(dist_path)

    return f"""<!DOCTYPE html>
<html>
<head><title>Neuro-Pulse Evaluation Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }}
table {{ border-collapse: collapse; width: 60%; margin: 20px 0; }}
th, td {{ border: 1px solid #ccc; padding: 8px 16px; text-align: left; }}
th {{ background-color: #1a5276; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
img {{ max-width: 100%; margin: 20px 0; }}
.pass {{ color: #2D7D4E; font-weight: bold; }}
.fail {{ color: #CC2222; font-weight: bold; }}
</style></head>
<body>
<h1>Neuro-Pulse Evaluation Report</h1>
<h2>Detection Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Accuracy</td><td>{metrics['accuracy']:.4f}</td></tr>
<tr><td>True Positive Rate</td><td>{metrics['tpr']:.4f}</td></tr>
<tr><td>True Negative Rate</td><td>{metrics['tnr']:.4f}</td></tr>
<tr><td>False Positive Rate</td><td>{metrics['fpr']:.4f}</td></tr>
<tr><td>False Negative Rate</td><td>{metrics['fnr']:.4f}</td></tr>
<tr><td>AUC</td><td>{metrics.get('auc', 0.0):.4f}</td></tr>
</table>
<h2>ROC Curve</h2>
<img src="data:image/png;base64,{roc_b64}" alt="ROC Curve">
<h2>SNR Distribution</h2>
<img src="data:image/png;base64,{dist_b64}" alt="SNR Distribution">
<hr>
<p><em>Neuro-Pulse | SRM AP University | CSE Section I</em></p>
</body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neuro-Pulse Evaluator — Generate Paper Figures & Tables"
    )
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, default="neuro_pulse/outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Neuro-Pulse Evaluator")
    print("=" * 50)

    dataset = load_dataset(args.real_dir, args.fake_dir)
    print(f"  Dataset: {len(dataset)} videos")

    metrics, per_file = run_evaluation(dataset, snr_threshold=args.threshold)

    print(f"\n  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics.get('auc', 0.0):.4f}")

    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    dist_path = os.path.join(args.output_dir, "snr_distribution.png")

    plot_roc_curve(per_file, roc_path)
    plot_snr_distribution(per_file, dist_path)

    tex = generate_latex_table(metrics)
    tex_path = os.path.join(args.output_dir, "results.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  LaTeX table saved to {tex_path}")

    html = generate_html_report(metrics, roc_path, dist_path)
    html_path = os.path.join(args.output_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  HTML report saved to {html_path}")

    print("\n" + "=" * 50)
    print("  Evaluation complete.")
    print("=" * 50)
