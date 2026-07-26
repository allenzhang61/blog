import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "forecast_comparison.svg"
TIMESFM_ENV = ROOT / ".venv_timesfm"
HORIZON = 12


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def ensure_main_packages() -> None:
    packages = [
        "chronos-forecasting==2.3.1",
        "gluonts==0.14.4",
        "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git@df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9",
        "matplotlib==3.11.1",
        "numpy==1.26.4",
        "pandas==2.1.4",
        "pyarrow==25.0.0",
        "setuptools==80.10.2",
        "torch==2.4.1",
        "transformers==4.57.6",
        "uni2ts==2.0.0",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *packages],
        env=clean_env(),
    )


def ensure_timesfm_env() -> Path:
    python = TIMESFM_ENV / "bin" / "python"
    if python.exists():
        return python

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to create the isolated TimesFM environment")

    subprocess.check_call([uv, "venv", str(TIMESFM_ENV), "--python", "3.12"], env=clean_env())
    subprocess.check_call(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "timesfm[torch]==2.0.2",
            "numpy==2.5.1",
        ],
        env=clean_env(),
    )
    return python


class ForecastResult(NamedTuple):
    name: str
    point: "np.ndarray"
    lower: "np.ndarray"
    upper: "np.ndarray"


def run_timesfm(context: "np.ndarray") -> ForecastResult:
    import numpy as np

    code = r"""
import json
import sys

import numpy as np
import torch
import timesfm

payload = json.loads(sys.stdin.read())
context = np.asarray(payload["context"], dtype=np.float64)

np.random.seed(0)
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=128,
        max_horizon=32,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
)
point, quantiles = model.forecast(horizon=12, inputs=[context])
print(json.dumps({
    "point": point[0].tolist(),
    "lower": quantiles[0, :, 0].tolist(),
    "upper": quantiles[0, :, -1].tolist(),
}))
"""
    completed = subprocess.run(
        [str(ensure_timesfm_env()), "-c", code],
        check=True,
        capture_output=True,
        input=json.dumps({"context": context.tolist()}),
        text=True,
        env=clean_env(),
        cwd=ROOT,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return ForecastResult(
        name="TimesFM 2.5",
        point=np.asarray(payload["point"]),
        lower=np.asarray(payload["lower"]),
        upper=np.asarray(payload["upper"]),
    )


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from chronos import Chronos2Pipeline
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation import make_evaluation_predictions
    from huggingface_hub import hf_hub_download
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    context = np.sin(np.linspace(0, 10, 96)) + np.linspace(0, 0.8, 96)

    def make_gluonts_dataset() -> ListDataset:
        return ListDataset(
            [
                {
                    "target": context.astype("float32"),
                    "start": pd.Period("2026-01-01", freq="D"),
                }
            ],
            freq="D",
        )

    def run_chronos() -> ForecastResult:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        context_df = pd.DataFrame(
            {
                "id": "series_0",
                "timestamp": pd.date_range("2026-01-01", periods=len(context), freq="D"),
                "target": context,
            }
        )
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            freq="D",
        )
        return ForecastResult(
            name="Chronos-2",
            point=pred_df["predictions"].to_numpy(),
            lower=pred_df["0.1"].to_numpy(),
            upper=pred_df["0.9"].to_numpy(),
        )

    def load_lag_llama_module(estimator: LagLlamaEstimator):
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            return estimator.create_lightning_module()
        finally:
            torch.load = original_load

    def run_lag_llama() -> ForecastResult:
        ckpt_path = hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=HORIZON,
            context_length=32,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=1,
            num_parallel_samples=50,
            device=torch.device("cpu"),
        )
        module = load_lag_llama_module(estimator)
        predictor = estimator.create_predictor(estimator.create_transformation(), module)
        forecast_it, _ = make_evaluation_predictions(
            dataset=make_gluonts_dataset(),
            predictor=predictor,
            num_samples=50,
        )
        forecast = next(forecast_it)
        return ForecastResult(
            name="Lag-Llama",
            point=np.asarray(forecast.mean),
            lower=np.asarray(forecast.quantile(0.1)),
            upper=np.asarray(forecast.quantile(0.9)),
        )

    def run_moirai() -> ForecastResult:
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
            prediction_length=HORIZON,
            context_length=len(context),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)
        forecast = next(iter(predictor.predict(make_gluonts_dataset())))
        return ForecastResult(
            name="Moirai 2 / Uni2TS",
            point=np.asarray(forecast.mean),
            lower=np.asarray(forecast.quantile(0.1)),
            upper=np.asarray(forecast.quantile(0.9)),
        )

    results = [
        run_timesfm(context),
        run_chronos(),
        run_lag_llama(),
        run_moirai(),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 12), sharex=True, layout="constrained")
    history_x = np.arange(len(context))
    forecast_x = np.arange(len(context), len(context) + HORIZON)

    for ax, result in zip(axes, results):
        ax.plot(history_x, context, color="#2563eb", linewidth=2, label="History")
        ax.plot(forecast_x, result.point, color="#dc2626", linewidth=2, marker="o", label=f"{result.name} forecast")
        ax.fill_between(forecast_x, result.lower, result.upper, color="#fca5a5", alpha=0.35, label="10%-90% interval")
        ax.axvline(len(context) - 1, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(f"{result.name} Forecast")
        ax.set_ylabel("Value")
        ax.legend()

    axes[-1].set_xlabel("Time step")
    fig.savefig(OUTPUT_PATH, format="svg")

    for result in results:
        print(f"{result.name}:")
        print(np.array2string(result.point, precision=3, separator=", "))
    print(f"Saved SVG: {OUTPUT_PATH}")


if __name__ == "__main__":
    if "--install" in sys.argv:
        ensure_main_packages()
    main()
