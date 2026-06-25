"""
1. stream swuggy from huggingface
2. perform ASR
3. Statistics on frequency-wise WER/CER/PER
"""

import polars as pl
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def num_all_samples():
    return len(pl.read_csv("hf://datasets/coml/sWuggy/inftrain/en/gold.csv"))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    data_files = {
        f"testset_{n}": f"hf://datasets/coml/sWuggy/inftrain/en/audio/testset_{n}-*.tar"
        for n in (1, 2, 4, 8, 16, 32, 64)
    }
    ds = load_dataset("webdataset", data_files=data_files, streaming=True)

    num_samples = num_all_samples()
    pbar = tqdm(total=num_samples, desc="ASR on Swuggy", unit="sample")
    with open("results.txt", "w") as f:
        f.write("key,asr,split_name\n")
        for split_name, split_ds in ds.items():
            for sample in split_ds:
                key = sample["__key__"]
                audio = sample["ogg"]  # bytes; decode with soundfile/librosa as needed
                result = pipe(audio)
                f.write(f"{key},{result['text'].strip()},{split_name}\n")
                pbar.update(1)


if __name__ == "__main__":
    main()
