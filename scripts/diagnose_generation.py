#!/usr/bin/env python
"""Generate one controlled sample and print audio health diagnostics.

Run from the repository root after installing dependencies and downloading a
model. This uses the same generation path as the Gradio app, so it is useful
for debugging Colab/T4 distortion without relying on browser state.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch
import torchaudio

from stable_audio_tools.interface import gradio as ui


DEFAULT_PROMPT = (
    "Synth Lead, Analog Synth, Warm, Clean, Focused, Simple Melody, Catchy, "
    "Low Reverb"
)


def first_model_pair(models_dir: Path) -> tuple[Path, Path]:
    ckpts = sorted(
        [
            *models_dir.rglob("*.safetensors"),
            *models_dir.rglob("*.ckpt"),
        ]
    )
    if not ckpts:
        raise FileNotFoundError(f"No .safetensors or .ckpt files found under {models_dir}")

    ckpt = ckpts[0]
    configs = sorted(ckpt.parent.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No model config JSON found next to {ckpt}")

    return ckpt, configs[0]


def tensor_stats(audio: torch.Tensor, sample_rate: int) -> dict[str, float | int]:
    audio = audio.to(torch.float32)
    total = int(audio.numel())
    finite = torch.isfinite(audio)
    nonfinite_count = int((~finite).sum().item())
    finite_audio = audio[finite] if finite.any() else torch.zeros(1)

    abs_audio = finite_audio.abs()
    peak = float(abs_audio.max().item())
    rms = float(torch.sqrt(torch.mean(finite_audio.square())).item())
    clipped = int((abs_audio >= 0.999).sum().item())

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[0]) if audio.ndim > 1 else 1,
        "samples": int(audio.shape[-1]),
        "duration_seconds": float(audio.shape[-1] / sample_rate),
        "total_values": total,
        "nonfinite_values": nonfinite_count,
        "peak_abs": peak,
        "rms": rms,
        "rms_dbfs": float(20 * math.log10(max(rms, 1e-12))),
        "clipped_values": clipped,
        "clipped_percent": float((clipped / max(total, 1)) * 100.0),
    }


def model_dtype_report() -> dict[str, str]:
    report = {}
    if ui.model is not None:
        report["model_first_param"] = str(next(ui.model.parameters()).dtype)
        if getattr(ui.model, "model", None) is not None:
            report["backbone_first_param"] = str(next(ui.model.model.parameters()).dtype)
        if getattr(ui.model, "pretransform", None) is not None:
            report["pretransform_first_param"] = str(next(ui.model.pretransform.parameters()).dtype)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Stable Audio diagnostic generation.")
    parser.add_argument("--config", default="config.json", help="Repo config JSON path.")
    parser.add_argument("--model-config", help="Model config JSON. Defaults to first downloaded model.")
    parser.add_argument("--ckpt-path", help="Checkpoint path. Defaults to first downloaded model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=128)
    parser.add_argument("--note", default="F")
    parser.add_argument("--scale", default="minor")
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde")
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--cfg-rescale", type=float, default=0.0)
    parser.add_argument("--skip-midi", action="store_true", help="Disable Basic Pitch during this run.")
    args = parser.parse_args()

    root = Path.cwd()
    with open(args.config) as f:
        repo_config = json.load(f)

    if args.ckpt_path and args.model_config:
        ckpt_path = Path(args.ckpt_path)
        model_config_path = Path(args.model_config)
    elif not args.ckpt_path and not args.model_config:
        ckpt_path, model_config_path = first_model_pair(root / repo_config["models_directory"])
    else:
        raise SystemExit("Pass both --ckpt-path and --model-config, or neither.")

    with open(model_config_path) as f:
        model_config = json.load(f)

    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    preferred_dtype = ui.pick_preferred_dtype(device)
    ui.DEVICE = device
    ui.PREFERRED_DTYPE = preferred_dtype

    print("== Environment ==")
    print(f"python_platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}")
    print(f"preferred_dtype: {preferred_dtype}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"model_config: {model_config_path}")

    ui.model, _ = ui.load_model(
        model_config=model_config,
        model_ckpt_path=str(ckpt_path),
        device=device,
        preferred_dtype=preferred_dtype,
    )

    print("== Model dtypes ==")
    print(json.dumps(model_dtype_report(), indent=2, sort_keys=True))

    if args.skip_midi:
        ui.convert_audio_to_midi = lambda audio_path, output_dir: None

    print("== Generating ==")
    output_audio, _spectrograms, piano_roll_path, midi_path = ui.generate_cond(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        bars=args.bars,
        bpm=args.bpm,
        note=args.note,
        scale=args.scale,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        seed=args.seed,
        sampler_type=args.sampler_type,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        cfg_rescale=args.cfg_rescale,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
    )

    audio, sample_rate = torchaudio.load(output_audio)
    stats = tensor_stats(audio, sample_rate)

    print("== Audio health ==")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print("== Outputs ==")
    print(f"wav: {output_audio}")
    print(f"midi: {midi_path}")
    print(f"piano_roll: {piano_roll_path}")

    if stats["nonfinite_values"] or stats["clipped_percent"] > 1.0 or stats["peak_abs"] >= 0.999:
        raise SystemExit("Diagnostic failed: generated audio appears numerically unhealthy.")

    print("Diagnostic passed: generated audio does not look clipped/NaN/Inf.")


if __name__ == "__main__":
    main()
