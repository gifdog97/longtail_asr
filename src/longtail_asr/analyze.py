"""
Analyze sWUGGY ASR outputs by word frequency.

Inputs:
- ASR output from longtail_asr.asr: key,asr,split_name
- sWUGGY gold.csv
- optionally, files under the sWUGGY frequencies directory

Outputs:
- per-item CER table
- frequency-bin mean CER table
- SVG plot with log frequency on the x axis and CER on the y axis
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ASRItem:
    key: str
    gold: str
    asr: str
    phones: str
    length: int
    frequency: int
    frequency_gold: int
    is_word: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sWUGGY ASR outputs by word frequency."
    )
    parser.add_argument(
        "--asr-output", type=str, required=True, help="Path to ASR output CSV file."
    )
    parser.add_argument(
        "--gold-csv", type=str, required=True, help="Path to sWUGGY gold CSV file."
    )
    parser.add_argument(
        "--frequency-dir",
        type=str,
        required=True,
        help="Path to frequency file directory.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        help="Path to save the plot PDF file.",
    )
    return parser.parse_args()


def load_frequency_data(frequency_dir: str) -> dict[str, int]:
    word_to_freq = {}
    for frequency_file in Path(frequency_dir).glob("*.csv"):
        with frequency_file.open("r", encoding="utf-8") as f:
            for line in f:
                line_split = line.strip().split(",")
                word, freq = line_split[0], line_split[1:]
                if not word:
                    continue
                word_to_freq[word] = int(sum(int(f) for f in freq))
    return word_to_freq


def load_gold_data(gold_csv: str) -> dict[str, dict[str, str | int]]:
    gold_data = {}
    with open(gold_csv, "r", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            key = row["filename"]
            frequency = row["frequency"] or 0
            gold_data[key] = {
                "word": row["word"],
                "frequency": int(frequency),
                "phones": row["phones"],
                "length": int(row["length"]),
            }
    return gold_data


def parse_asr_output(
    asr_output_path: str,
    gold_data: dict[str, dict[str, str | int]],
    frequency_data: dict[str, int],
) -> list[ASRItem]:
    asr_items = []
    with open(asr_output_path, "r", encoding="utf-8") as f:
        for line in f:
            # need adhoc splitting as `,` can appear in the ASR output...
            line_split = line.strip().split(",")
            key, asr, _ = (
                line_split[0],
                "".join(line_split[1:-1]),
                line_split[-1],
            )
            if key == "key":
                continue
            item_gold = gold_data.get(key, {})
            word_gold = item_gold.get("word", "")
            frequency_gold = item_gold.get("frequency", 0)
            phones = item_gold.get("phones", "")
            length = int(item_gold.get("length", 0))
            is_word = word_gold != ""
            frequency = frequency_data.get(word_gold, 0) if is_word else 0
            asr_item = ASRItem(
                key=key,
                gold=word_gold,
                asr=asr,
                phones=phones,
                length=length,
                frequency=frequency,
                frequency_gold=frequency_gold if is_word else 0,
                is_word=is_word,
            )
            asr_items.append(asr_item)
    return asr_items


def group_items(asr_items: list[ASRItem]) -> dict[str, dict[str, str | list[str]]]:
    # initial implementation: word-level
    grouped_data = {}
    for item in asr_items:
        if not item.is_word:
            continue
        word = item.gold
        if word not in grouped_data:
            grouped_data[word] = {
                "asr": [],
                "frequency": item.frequency,
                "frequency_gold": item.frequency_gold,
            }
        grouped_data[word]["asr"].append(item.asr)
    for word, data in grouped_data.items():
        cer = jiwer.cer(data["asr"], [word] * len(data["asr"]))
        data["cer"] = cer
    return grouped_data


def plot_logfreq_cer(
    grouped_data: dict[str, dict[str, str | list[str]]], plot_path: str
) -> None:
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    frequencies = []
    cers = []
    for word, data in grouped_data.items():
        frequencies.append(data["frequency"])
        cers.append(data["cer"])
    log_frequencies = np.log10(np.array(frequencies))
    plt.scatter(log_frequencies, cers)
    plt.xlabel("Log Frequency")
    plt.ylabel("Character Error Rate (CER)")
    plt.title("Log Frequency vs CER")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    args = parse_args()
    frequency_data = load_frequency_data(args.frequency_dir)
    gold_data = load_gold_data(args.gold_csv)
    asr_items = parse_asr_output(args.asr_output, gold_data, frequency_data)
    grouped_data = group_items(asr_items)
    plot_logfreq_cer(grouped_data, args.plot_path)


if __name__ == "__main__":
    main()
