"""
Analyze sWUGGY ASR outputs by word frequency.

Inputs:
- ASR output directory from longtail_asr.asr: key,asr,split_name
- sWUGGY gold.csv
- optionally, files under the sWUGGY frequencies directory

Outputs:
- per-item CER table
- frequency-bin CER summary plot (median + IQR band)
- small-multiples plot by ASR output
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


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


ASRStats = dict[str, dict[str, str | int | float | list[str]]]
PlotPoints = dict[str, tuple[np.ndarray, np.ndarray]]
NUM_BINS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sWUGGY ASR outputs by word frequency."
    )
    parser.add_argument(
        "--asr-output-dir",
        type=str,
        required=True,
        help="Path to ASR output CSV directory.",
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
        required=True,
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


def get_word_to_stats(asr_items: list[ASRItem]) -> ASRStats:
    word_to_stats = {}
    for item in asr_items:
        if not item.is_word:
            continue
        word = item.gold
        if word not in word_to_stats:
            word_to_stats[word] = {
                "asr": [],
                "frequency": item.frequency,
                "frequency_gold": item.frequency_gold,
            }
        word_to_stats[word]["asr"].append(item.asr)
    for word, data in word_to_stats.items():
        cer = jiwer.cer(data["asr"], [word] * len(data["asr"]))
        data["cer"] = cer
    return word_to_stats


def arrange_asr_stats(
    asr_output_dir: str,
    gold_data: dict[str, dict[str, str | int]],
    frequency_data: dict[str, int],
) -> dict[str, ASRStats]:
    asr_stats = {}
    for asr_output_path in Path(asr_output_dir).glob("*.csv"):
        asr_items = parse_asr_output(str(asr_output_path), gold_data, frequency_data)
        asr_stats[asr_output_path.name] = get_word_to_stats(asr_items)
    return asr_stats


def get_plot_points(asr_stats: ASRStats) -> tuple[np.ndarray, np.ndarray]:
    frequencies = []
    cers = []
    for data in asr_stats.values():
        frequency = int(data["frequency"])
        if frequency <= 0:
            continue
        frequencies.append(frequency)
        cers.append(float(data["cer"]))
    return np.log10(np.array(frequencies)), np.array(cers)


def get_all_plot_points(asr_stats: dict[str, ASRStats]) -> PlotPoints:
    # label (ASR system name) to PlotPoints (x=log_frequencies, y=cers)
    return {
        label: get_plot_points(grouped_data)
        for label, grouped_data in asr_stats.items()
    }


def get_bin_edges(plot_points: PlotPoints, num_bins: int = NUM_BINS) -> np.ndarray:
    all_log_frequencies = [
        log_frequencies
        for log_frequencies, _ in plot_points.values()
        if len(log_frequencies) > 0
    ]
    if not all_log_frequencies:
        raise ValueError("No plottable points found.")
    values = np.concatenate(all_log_frequencies)
    return np.linspace(values.min(), values.max(), num_bins + 1)


def calculate_bin_stats(
    log_frequencies: np.ndarray, cers: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    medians = np.full(len(bin_centers), np.nan)
    q25s = np.full(len(bin_centers), np.nan)
    q75s = np.full(len(bin_centers), np.nan)

    for idx in range(len(bin_centers)):
        if idx == len(bin_centers) - 1:
            in_bin = (bin_edges[idx] <= log_frequencies) & (
                log_frequencies <= bin_edges[idx + 1]
            )
        else:
            in_bin = (bin_edges[idx] <= log_frequencies) & (
                log_frequencies < bin_edges[idx + 1]
            )
        bin_cers = cers[in_bin]
        if len(bin_cers) == 0:
            continue
        q25s[idx], medians[idx], q75s[idx] = np.percentile(bin_cers, [25, 50, 75])

    return bin_centers, medians, q25s, q75s


def draw_model_summary(
    ax: Axes,
    log_frequencies: np.ndarray,
    cers: np.ndarray,
    bin_edges: np.ndarray,
    color: str,
    label: str | None = None,
) -> None:
    if len(cers) == 0:
        return

    bin_centers, medians, q25s, q75s = calculate_bin_stats(
        log_frequencies, cers, bin_edges
    )
    ax.fill_between(bin_centers, q25s, q75s, color=color, alpha=0.12, linewidth=0)
    ax.plot(bin_centers, medians, color=color, linewidth=2.0, label=label)


def plot_logfreq_cer(asr_stats: dict[str, ASRStats], plot_path: str) -> None:
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_points = get_all_plot_points(asr_stats)
    bin_edges = get_bin_edges(plot_points)
    colors = plt.get_cmap("tab20").colors

    for idx, (label, (log_frequencies, cers)) in enumerate(plot_points.items()):
        draw_model_summary(
            ax,
            log_frequencies,
            cers,
            bin_edges,
            colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel("Log Frequency")
    ax.set_ylabel("Character Error Rate (CER)")
    ax.set_title("Log Frequency vs CER")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency_data = load_frequency_data(args.frequency_dir)
    gold_data = load_gold_data(args.gold_csv)
    asr_stats = arrange_asr_stats(args.asr_output_dir, gold_data, frequency_data)
    plot_logfreq_cer(asr_stats, args.plot_path)


if __name__ == "__main__":
    main()
