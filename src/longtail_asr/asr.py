"""
Run ASR over sWUGGY and write one CSV-like result file per model.

The default command starts one worker per visible CUDA device. Each worker
loads the requested model, shards each Hugging Face Dataset split by
rank/world_size, runs batched ASR, and writes a temporary part file. The parent
process merges those part files into the final model-named output file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

DEFAULT_MODEL_IDS = (
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "facebook/seamless-m4t-v2-large",
    "facebook/mms-1b-all",
    "espnet/owsm_ctc_v4_1B",
    "espnet/owsm_v4_medium_1B",
    "facebook/hubert-large-ls960-ft",
)

SWUGGY_TESTSETS = (1, 2, 4, 8, 16, 32, 64)
LANGUAGE_TO_MMS = {"en": "eng", "fr": "fra"}
LANGUAGE_TO_WHISPER = {"en": "english", "fr": "french"}
LANGUAGE_TO_SEAMLESS = {"en": "eng", "fr": "fra"}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str


MODEL_SPECS = {
    "openai/whisper-large-v3": ModelSpec("openai/whisper-large-v3", "whisper"),
    "openai/whisper-large-v3-turbo": ModelSpec(
        "openai/whisper-large-v3-turbo", "whisper"
    ),
    "facebook/seamless-m4t-v2-large": ModelSpec(
        "facebook/seamless-m4t-v2-large", "hf_pipeline"
    ),
    "facebook/mms-1b-all": ModelSpec("facebook/mms-1b-all", "mms_ctc"),
    "espnet/owsm_ctc_v4_1B": ModelSpec("espnet/owsm_ctc_v4_1B", "hf_pipeline"),
    "espnet/owsm_v4_medium_1B": ModelSpec("espnet/owsm_v4_medium_1B", "hf_pipeline"),
    "facebook/hubert-large-ls960-ft": ModelSpec(
        "facebook/hubert-large-ls960-ft", "hf_pipeline"
    ),
}


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


def dtype_for_device(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def get_model_spec(model_id: str, backend: str) -> ModelSpec:
    if backend != "auto":
        return ModelSpec(model_id, backend)
    if model_id in MODEL_SPECS:
        return MODEL_SPECS[model_id]
    if "whisper" in model_id:
        return ModelSpec(model_id, "whisper")
    if "mms-1b" in model_id:
        return ModelSpec(model_id, "mms_ctc")
    return ModelSpec(model_id, "hf_pipeline")


def build_asr_pipeline(
    model_id: str,
    *,
    backend: str,
    device: str,
    language: str,
) -> tuple[Any, dict[str, Any]]:
    spec = get_model_spec(model_id, backend)
    dtype = dtype_for_device(device)

    if spec.backend == "whisper":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
        )
        generate_kwargs = {
            "task": "transcribe",
            "language": LANGUAGE_TO_WHISPER.get(language, language),
        }
        return pipe, {"generate_kwargs": generate_kwargs}

    if spec.backend == "mms_ctc":
        target_lang = LANGUAGE_TO_MMS.get(language, language)
        processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
        model = AutoModelForCTC.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(model, "load_adapter"):
            model.load_adapter(target_lang)
        if hasattr(processor, "tokenizer") and hasattr(
            processor.tokenizer, "set_target_lang"
        ):
            processor.tokenizer.set_target_lang(target_lang)
        model.to(device)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
        )
        return pipe, {}

    if spec.backend == "hf_pipeline":
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            dtype=dtype,
            device=device,
            trust_remote_code=True,
        )
        generate_kwargs: dict[str, Any] = {}
        if "seamless-m4t" in model_id:
            generate_kwargs["tgt_lang"] = LANGUAGE_TO_SEAMLESS.get(language, language)
        return pipe, {"generate_kwargs": generate_kwargs} if generate_kwargs else {}

    msg = (
        f"Unsupported backend '{spec.backend}'. "
        "Use one of: auto, whisper, mms_ctc, hf_pipeline."
    )
    raise ValueError(msg)


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
    pipe, call_kwargs = build_asr_pipeline(
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
        desc=f"ASR rank {args.rank}/{args.world_size} on {Path(args.part_file).name}",
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
                results = pipe(
                    audios,
                    batch_size=args.batch_size,
                    **call_kwargs,
                )
                if isinstance(results, dict):
                    results = [results]
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
        raise RuntimeError(f"ASR worker failed ({details}).")

    merge_part_files(part_files, output_file)
    print(f"Wrote {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batched ASR on sWUGGY.")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run ASR and merge worker outputs.")
    run.add_argument("--model-id", required=True)
    run.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "whisper", "mms_ctc", "hf_pipeline"),
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
        choices=("auto", "whisper", "mms_ctc", "hf_pipeline"),
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
