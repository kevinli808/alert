"""Evaluate detection accuracy on labelled recordings and pick thresholds.

Layout (one folder per app category id; "background" for sounds that must NOT alert):
    eval_data/
      fire_alarm/*.wav   doorbell/*.wav   baby_crying/*.wav   background/*.wav ...

Run:   python -m scripts.evaluate_model eval_data [--mlflow]
Prints precision / recall per category at several thresholds. Use the results to set
CONFIDENCE_THRESHOLD and CATEGORY_THRESHOLDS. With --mlflow, metrics are logged to the active
MLflow tracking server (e.g. Databricks: set DATABRICKS_HOST/TOKEN and MLFLOW_TRACKING_URI=databricks).
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

from app.core.errors import AppError
from app.services.audio_processor import AudioProcessor
from app.services.sound_categories import CATEGORY_IDS
from app.services.sound_classifier import YamnetClassifier

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=pathlib.Path)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    proc = AudioProcessor(max_bytes=50_000_000, min_seconds=0.2, max_seconds=60, silence_rms_threshold=0.0,
                          ffmpeg_path="auto")
    clf = YamnetClassifier("https://tfhub.dev/google/yamnet/1")
    clf.load()

    samples: list[tuple[str, dict[str, float]]] = []
    for folder in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        label = folder.name
        if label != "background" and label not in CATEGORY_IDS:
            print(f"skip unknown folder {label}")
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            try:
                audio = proc.process(f.read_bytes())
                samples.append((label, clf.classify(audio.waveform).category_scores))
            except AppError as exc:
                print(f"skip {f}: {exc.message}")
    print(f"{len(samples)} clips\n")

    metrics: dict[str, float] = {}
    present = sorted({lbl for lbl, _ in samples if lbl != "background"})
    for cat in present:
        print(f"== {cat}")
        for th in THRESHOLDS:
            tp = sum(1 for lbl, s in samples if lbl == cat and s[cat] >= th)
            fn = sum(1 for lbl, s in samples if lbl == cat and s[cat] < th)
            fp = sum(1 for lbl, s in samples if lbl != cat and s[cat] >= th)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            print(f"  th={th:.2f}  precision={prec:.2f}  recall={rec:.2f}  (tp={tp} fp={fp} fn={fn})")
            metrics[f"{cat}_p_{int(th*100)}"] = prec
            metrics[f"{cat}_r_{int(th*100)}"] = rec

    if args.mlflow:
        import mlflow

        with mlflow.start_run(run_name="yamnet-threshold-eval"):
            mlflow.log_param("clips", len(samples))
            mlflow.log_metrics(metrics)
        print("logged to MLflow")


if __name__ == "__main__":
    main()
