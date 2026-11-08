"""Package YAMNet + the app's sound categories as an MLflow pyfunc, register it in Unity Catalog and
serve it on Databricks Model Serving.

The YAMNet SavedModel (~18 MB) is bundled as a model artifact, so the serving container never has to
download it from tfhub.dev at start-up.

Run locally (MLflow + databricks-sdk installed, TensorFlow NOT required), from backend/:

    set DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
    set DATABRICKS_TOKEN=<personal access token>
    python databricks/yamnet_pyfunc.py --yamnet-dir <path to YAMNet SavedModel> \
        --model workspace.alert.yamnet_sound_alerts --endpoint yamnet-sound-alerts

<path to YAMNet SavedModel> is the folder holding saved_model.pb (tensorflow_hub caches it under
%TEMP%/tfhub_modules/<hash> after the backend has loaded YAMNet once).

Contract matches app/services/databricks_service.py:
  request  {"dataframe_records": [{"audio_b64": <base64 little-endian int16 PCM, mono, 16 kHz>,
                                   "sample_rate": 16000}]}
  response {"predictions": [{"category_scores": {...}, "top_labels": [[label, score], ...]}]}
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import mlflow
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CATEGORIES_FILE = os.path.normpath(os.path.join(HERE, "..", "app", "services", "sound_categories.py"))
PIP_REQUIREMENTS = ["tensorflow==2.20.*", "tf-keras==2.20.*", "tensorflow-hub==0.16.1", "numpy<2.5", "pandas"]


class YamnetSoundAlerts(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import csv

        import tensorflow as tf
        import tensorflow_hub as hub

        import sound_categories as sc  # shipped via code_paths

        self.sc = sc
        source = context.artifacts.get("yamnet") or context.model_config.get(
            "yamnet_handle", "https://tfhub.dev/google/yamnet/1"
        )
        self.model = hub.load(source)
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
            waveform = pcm.astype(np.float32) / 32768.0
            scores, _, _ = self.model(waveform)
            per_class = self.sc.clip_scores(scores.numpy(), self.aggregation)
            out.append(
                {
                    "category_scores": self.sc.category_scores(per_class, self.label_index),
                    "top_labels": [[l, s] for l, s in self.sc.top_labels(per_class, self.class_names, 3)],
                }
            )
        return out


def _signature():
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Array, ColSpec, Map, Object, Property, Schema

    inputs = Schema([ColSpec("string", "audio_b64"), ColSpec("long", "sample_rate")])
    outputs = Schema(
        [
            ColSpec(Map("double"), "category_scores"),
            ColSpec(Array("string"), "top_labels"),  # [[label, score], ...]: informational only
        ]
    )
    return ModelSignature(inputs=inputs, outputs=outputs)


def log_and_register(model_name: str, yamnet_dir: str, experiment: str | None = None) -> str:
    """Log the pyfunc with the bundled SavedModel and register it. Returns the new version number."""
    if not os.path.isfile(os.path.join(yamnet_dir, "saved_model.pb")):
        raise SystemExit(f"{yamnet_dir} does not contain saved_model.pb")
    mlflow.set_tracking_uri("databricks")
    if "." in model_name:
        mlflow.set_registry_uri("databricks-uc")
    if experiment:
        mlflow.set_experiment(experiment)
    sys.path.insert(0, os.path.dirname(CATEGORIES_FILE))
    with mlflow.start_run(run_name="yamnet-sound-alerts"):
        mlflow.log_params({"base_model": "google/yamnet/1", "aggregation": "max"})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=YamnetSoundAlerts(),
            artifacts={"yamnet": yamnet_dir},
            code_paths=[CATEGORIES_FILE],
            model_config={"aggregation": "max"},
            signature=_signature(),
            pip_requirements=PIP_REQUIREMENTS,
            registered_model_name=model_name,
        )
    return str(info.registered_model_version)


def deploy_endpoint(endpoint: str, model_name: str, version: str, scale_to_zero: bool = True) -> None:
    """Create the serving endpoint, or point an existing one at the new version."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    w = WorkspaceClient()
    entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=scale_to_zero,
    )
    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint in existing:
        w.serving_endpoints.update_config(name=endpoint, served_entities=[entity])
    else:
        w.serving_endpoints.create(
            name=endpoint, config=EndpointCoreConfigInput(name=endpoint, served_entities=[entity])
        )
    print(f"Endpoint {endpoint!r} is building version {version} (this usually takes 10-20 minutes).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yamnet-dir", required=True)
    parser.add_argument("--model", default="workspace.alert.yamnet_sound_alerts")
    parser.add_argument("--endpoint", default="yamnet-sound-alerts")
    parser.add_argument("--experiment", default=None, help="e.g. /Users/<you>/yamnet-sound-alerts")
    parser.add_argument("--no-scale-to-zero", action="store_true")
    parser.add_argument("--skip-endpoint", action="store_true")
    args = parser.parse_args()
    version = log_and_register(args.model, args.yamnet_dir, args.experiment)
    print(f"Registered {args.model} version {version}")
    if not args.skip_endpoint:
        deploy_endpoint(args.endpoint, args.model, version, scale_to_zero=not args.no_scale_to_zero)


if __name__ == "__main__":
    main()
