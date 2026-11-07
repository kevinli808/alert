"""Log YAMNet + the app's category mapping as an MLflow pyfunc and register it for Databricks
Model Serving. Run this in a Databricks notebook (or locally with MLflow pointed at Databricks).

    %pip install tensorflow==2.20.* tensorflow-hub==0.16.1 mlflow numpy
    # upload this folder's parent (backend/) to your workspace / Repos, then:
    %run ./yamnet_pyfunc

After registering, create a serving endpoint (Serving → Create endpoint → choose the registered
model, CPU, "Small", scale-to-zero ON for a hackathon) and set DATABRICKS_ENDPOINT_NAME.

Contract matches app/services/databricks_service.py:
  input  DataFrame columns: audio_b64 (base64 little-endian int16 PCM mono 16 kHz), sample_rate
  output one dict per row: {"category_scores": {...}, "top_labels": [[label, score], ...]}
"""

from __future__ import annotations

import base64
import os
import sys

import mlflow
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CATEGORIES_FILE = os.path.join(HERE, "..", "app", "services", "sound_categories.py")


class YamnetSoundAlerts(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import csv

        import tensorflow as tf
        import tensorflow_hub as hub

        import sound_categories as sc  # shipped via code_paths

        self.sc = sc
        self.model = hub.load(context.model_config.get("yamnet_handle", "https://tfhub.dev/google/yamnet/1"))
        path = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(path) as fh:
            self.class_names = [r["display_name"] for r in csv.DictReader(fh)]
        self.label_index, _ = sc.build_label_index(self.class_names)
        self.aggregation = context.model_config.get("aggregation", "max")

    def predict(self, context, model_input: pd.DataFrame, params=None):
        out = []
        for _, row in model_input.iterrows():
            if int(row.get("sample_rate", 16000)) != 16000:
                raise ValueError("sample_rate must be 16000")
            pcm = np.frombuffer(base64.b64decode(row["audio_b64"]), dtype="<i2")
            waveform = (pcm.astype(np.float32) / 32768.0)
            scores, _, _ = self.model(waveform)
            per_class = self.sc.clip_scores(scores.numpy(), self.aggregation)
            out.append({
                "category_scores": self.sc.category_scores(per_class, self.label_index),
                "top_labels": [[l, s] for l, s in self.sc.top_labels(per_class, self.class_names, 3)],
            })
        return out


def log_and_register(model_name: str = "yamnet_sound_alerts") -> None:
    sys.path.insert(0, os.path.dirname(CATEGORIES_FILE))
    example = pd.DataFrame([{"audio_b64": base64.b64encode(np.zeros(16000, "<i2").tobytes()).decode(), "sample_rate": 16000}])
    # Unity Catalog: use "catalog.schema.yamnet_sound_alerts" and mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name="yamnet-baseline"):
        mlflow.log_params({"base_model": "google/yamnet/1", "aggregation": "max"})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YamnetSoundAlerts(),
            code_paths=[CATEGORIES_FILE],
            model_config={"yamnet_handle": "https://tfhub.dev/google/yamnet/1", "aggregation": "max"},
            input_example=example,
            pip_requirements=["tensorflow==2.20.*", "tensorflow-hub==0.16.1", "numpy", "pandas"],
            registered_model_name=model_name,
        )


if __name__ == "__main__":
    log_and_register()
