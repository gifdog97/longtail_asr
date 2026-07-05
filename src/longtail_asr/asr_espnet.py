"""
Run ESPnet ASR over sWUGGY and write one CSV-like result file per model.

This is intentionally separate from asr.py so the ESPnet pixi environment does
not need the Hugging Face Transformers ASR stack.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import soundfile as sf
import torch
from datasets import load_dataset
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
from tqdm import tqdm

DEFAULT_MODEL_IDS = (
    "espnet/owsm_ctc_v4_1B",
    "espnet/owsm_v4_medium_1B",
)
SWUGGY_TESTSETS = (1, 2, 4, 8, 16, 32, 64)
LANGUAGE_TO_ESPNET = {"en": "<eng>", "fr": "<fra>"}


class EspnetCTCASR:
    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        language: str,
        context_len_in_secs: int = 4,
    ) -> None:
        self.s2t = Speech2TextGreedySearch.from_pretrained(
            model_id,
            device=device,
            use_flash_attn=False,
            lang_sym=espnet_language_symbol(language),
            task_sym="<asr>",
        )
        self.context_len_in_secs = context_len_in_secs

    def __call__(self, audios: list[Any], *, batch_size: int) -> list[dict[str, str]]:
        inputs = [espnet_audio_input(audio) for audio in audios]
        results = self.s2t.batch_decode(
            inputs,
            batch_size=batch_size,
            context_len_in_secs=self.context_len_in_secs,
        )
        if isinstance(results, str):
            results = [results]
        return [{"text": text} for text in results]


class EspnetS2TASR:
    def __init__(self, model_id: str, *, device: str, language: str) -> None:
        self.s2t = Speech2Text.from_pretrained(
            model_tag=model_id,
            device=device,
            beam_size=5,
            ctc_weight=0.0,
            maxlenratio=0.0,
            lang_sym=espnet_language_symbol(language),
            task_sym="<asr>",
            predict_time=False,
        )

    def __call__(self, audios: list[Any], *, batch_size: int) -> list[dict[str, str]]:
        del batch_size
        return [{"text": self.decode_one(audio)} for audio in audios]

    def decode_one(self, audio: Any) -> str:
        result = self.s2t(espnet_audio_input(audio))[0]
        return str(result[-2])


def num_all_samples(language: str) -> int:
    return len(pl.read_csv(f"hf://datasets/coml/sWuggy/inftrain/{language}/gold.csv"))


def swuggy_data_files(language: str) -> dict[str, str]:
    return {
        f"testset_{n}": (
            f"hf://datasets/coml/sWuggy/inftrain/{language}/audio/testset_{n}-*.tar"
        )
        for n in SWUGGY_TESTSETS
    }


def load_swuggy(language: str, *, streaming: bool) -> Any:
    return load_dataset(
        "webdataset",
        data_files=swuggy_data_files(language),
        streaming=streaming,
    )


def safe_model_filename(model_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("_")
    return f"{normalized}.csv"


def espnet_language_symbol(language: str) -> str:
    if language.startswith("<") and language.endswith(">"):
        return language
    return LANGUAGE_TO_ESPNET.get(language, f"<{language}>")


def espnet_audio_input(audio: Any) -> Any:
    speech, _ = read_audio(audio)
    return speech


def read_audio(audio: Any) -> tuple[Any, int | None]:
    if hasattr(audio, "get_all_samples"):
        return read_torchcodec_audio(audio)

    if isinstance(audio, dict):
        if audio.get("path"):
            return read_soundfile(audio["path"])
        if "bytes" in audio:
            return read_soundfile(io.BytesIO(audio["bytes"]))
        if "raw" in audio and isinstance(audio["raw"], bytes):
            return read_soundfile(io.BytesIO(audio["raw"]))
        if "array" in audio:
            return normalize_speech_array(audio["array"]), audio.get("sampling_rate")
        if "raw" in audio:
            return normalize_speech_array(audio["raw"]), audio.get("sampling_rate")

    if isinstance(audio, (str, os.PathLike)):
        return read_soundfile(audio)
    if isinstance(audio, bytes):
        return read_soundfile(io.BytesIO(audio))
    return normalize_speech_array(audio), None


def read_torchcodec_audio(audio: Any) -> tuple[Any, int | None]:
    samples = audio.get_all_samples()
    speech = normalize_speech_array(samples.data, channel_first=True)
    rate = getattr(samples, "sample_rate", None)
    if rate is None and hasattr(audio, "metadata"):
        rate = getattr(audio.metadata, "sample_rate", None)
    return speech, rate


def read_soundfile(file: Any) -> tuple[Any, int]:
    speech, rate = sf.read(file, dtype="float32")
    return normalize_speech_array(speech), rate


def normalize_speech_array(audio: Any, *, channel_first: bool = False) -> Any:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    speech = np.asarray(audio, dtype="float32")
    if speech.ndim == 0:
        raise ValueError("Expected audio samples, got a scalar.")

    if speech.ndim > 2:
        speech = np.squeeze(speech)
    if speech.ndim == 2:
        if channel_first:
            speech = speech.mean(axis=0)
        else:
            speech = speech.mean(axis=1)
    elif speech.ndim != 1:
        raise ValueError(f"Expected 1D or 2D audio samples, got shape {speech.shape}.")

    return np.ascontiguousarray(speech, dtype=np.float32)


def get_backend(model_id: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if "ctc" in model_id:
        return "espnet_ctc"
    return "espnet_s2t"


def build_asr(
    model_id: str,
    *,
    backend: str,
    device: str,
    language: str,
) -> Any:
    backend = get_backend(model_id, backend)
    if backend == "espnet_ctc":
        return EspnetCTCASR(model_id, device=device, language=language)
    if backend == "espnet_s2t":
        return EspnetS2TASR(model_id, device=device, language=language)
    raise ValueError(
        f"Unsupported backend '{backend}'. Use one of: auto, espnet_ctc, espnet_s2t."
    )


def batch_rows(
    batch: dict[str, list[Any]], split_name: str
) -> tuple[list[str], list[Any]]:
    keys = [str(key) for key in batch["__key__"]]
    if "ogg" not in batch:
        raise KeyError(f"Expected an 'ogg' audio column in split {split_name}.")
    return keys, list(batch["ogg"])


def iter_dataset_batches(
    dataset: Any, batch_size: int
) -> Iterable[dict[str, list[Any]]]:
    if hasattr(dataset, "iter"):
        yield from dataset.iter(batch_size=batch_size)
        return

    batch: dict[str, list[Any]] = {}
    for sample in dataset:
        for key, value in sample.items():
            batch.setdefault(key, []).append(value)
        if len(next(iter(batch.values()))) >= batch_size:
            yield batch
            batch = {}
    if batch:
        yield batch


def shard_dataset(dataset: Any, *, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return dataset
    if not hasattr(dataset, "shard"):
        raise TypeError("Loaded dataset does not support sharding.")
    return dataset.shard(num_shards=world_size, index=rank)


def run_worker(args: argparse.Namespace) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    asr = build_asr(
        args.model_id,
        backend=args.backend,
        device=device,
        language=args.language,
    )
    ds = load_swuggy(args.language, streaming=args.streaming)

    total = None
    if args.show_total:
        total = max(1, num_all_samples(args.language) // args.world_size)

    progress = tqdm(
        total=total,
        desc=f"ESPnet ASR rank {args.rank}/{args.world_size} on {Path(args.part_file).name}",
        unit="sample",
    )
    with open(args.part_file, "w") as f:
        for split_name, split_ds in ds.items():
            worker_ds = shard_dataset(
                split_ds,
                rank=args.rank,
                world_size=args.world_size,
            )
            for batch in iter_dataset_batches(worker_ds, args.batch_size):
                keys, audios = batch_rows(batch, split_name)
                results = asr(audios, batch_size=args.batch_size)
                for key, result in zip(keys, results, strict=True):
                    f.write(f"{key},{result['text'].strip()},{split_name}\n")
                progress.update(len(keys))
    progress.close()


def merge_part_files(part_files: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as out:
        out.write("key,asr,split_name\n")
        for part_file in part_files:
            with open(part_file) as part:
                for line in part:
                    out.write(line)


def visible_cuda_devices() -> list[str]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [device.strip() for device in cuda_visible.split(",") if device.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def infer_world_size(requested: int | None) -> int:
    if requested is not None:
        return requested
    return max(1, len(visible_cuda_devices()))


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_file = (
        Path(args.output_file)
        if args.output_file
        else output_dir / safe_model_filename(args.model_id)
    )
    tmp_dir = output_dir / ".parts" / output_file.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    world_size = infer_world_size(args.num_workers)
    if args.devices:
        devices = args.devices.split(",")
        world_size = len(devices)
    elif world_size == 1 and not torch.cuda.is_available():
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(world_size)]

    part_files = [tmp_dir / f"rank_{rank:03d}.csv" for rank in range(world_size)]
    commands = []
    for rank, part_file in enumerate(part_files):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-id",
            args.model_id,
            "--backend",
            args.backend,
            "--language",
            args.language,
            "--batch-size",
            str(args.batch_size),
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--device",
            devices[rank],
            "--part-file",
            str(part_file),
        ]
        if args.streaming:
            command.append("--streaming")
        if args.show_total:
            command.append("--show-total")
        commands.append(command)

    env = os.environ.copy()
    procs = [subprocess.Popen(command, env=env) for command in commands]
    failures = []
    for rank, proc in enumerate(procs):
        returncode = proc.wait()
        if returncode != 0:
            failures.append((rank, returncode))
    if failures:
        details = ", ".join(f"rank {rank}: {code}" for rank, code in failures)
        raise RuntimeError(f"ESPnet ASR worker failed ({details}).")

    merge_part_files(part_files, output_file)
    print(f"Wrote {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ESPnet ASR on sWUGGY.")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run ASR and merge worker outputs.")
    run.add_argument("--model-id", required=True)
    run.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "espnet_ctc", "espnet_s2t"),
    )
    run.add_argument("--language", default="en", choices=("en", "fr"))
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--num-workers", type=int)
    run.add_argument("--devices", help="Comma-separated devices, e.g. cuda:0,cuda:1")
    run.add_argument("--output-dir", default="results")
    run.add_argument("--output-file")
    run.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming datasets. Default uses Hugging Face Dataset objects.",
    )
    run.add_argument("--show-total", action="store_true")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--model-id", required=True)
    worker.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "espnet_ctc", "espnet_s2t"),
    )
    worker.add_argument("--language", default="en", choices=("en", "fr"))
    worker.add_argument("--batch-size", type=int, default=16)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--part-file", required=True)
    worker.add_argument("--streaming", action="store_true")
    worker.add_argument("--show-total", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    if args.command == "worker":
        run_worker(args)
        return
    if args.command == "run":
        run_parent(args)
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
