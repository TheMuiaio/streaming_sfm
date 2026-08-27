#!/usr/bin/env python3
"""Prepare MCIF scoring inputs for OmniSTEval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def _resolve(path: Path) -> Path:
    return path.resolve()


def _load_yaml_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {path}")
    return data


def _seg_counts_from_audio_yaml(entries: list[dict], file_order: list[str]) -> dict[str, int]:
    raw_counts = Counter(Path(entry["wav"]).stem for entry in entries)
    missing = [talk for talk in file_order if talk not in raw_counts]
    if missing:
        raise ValueError(f"Missing segment counts for talks: {', '.join(missing[:5])}")
    return {talk: raw_counts[talk] for talk in file_order}


def split_reference_file(
    ref_path: Path,
    file_order: list[str],
    seg_counts: dict[str, int],
    out_dir: Path,
    suffix: str,
) -> list[Path]:
    lines = [line.rstrip("\n") for line in ref_path.read_text(encoding="utf-8").splitlines()]
    expected = sum(seg_counts[talk] for talk in file_order)
    if len(lines) != expected:
        raise ValueError(
            f"{ref_path} has {len(lines)} lines, expected {expected} from audio-segments.yaml"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[Path] = []
    idx = 0
    for talk in file_order:
        n = seg_counts[talk]
        chunk = lines[idx : idx + n]
        idx += n
        out_file = out_dir / f"{talk}{suffix}"
        out_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        out_files.append(out_file)
    return out_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcif-root",
        type=Path,
        default=Path.home() / ".cache/simuleval/mcif_iwslt26",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="mcif_set",
        default="test",
        help="MCIF split name (default: test).",
    )
    args = parser.parse_args()

    mcif_root = _resolve(args.mcif_root)
    output_dir = _resolve(args.output_dir)
    mcif_set = args.mcif_set
    scoring_dir = output_dir / "scoring_data"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    file_order_path = mcif_root / f"en-de_{mcif_set}_wavs_list.txt"
    file_order = [
        Path(line.strip()).stem
        for line in file_order_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    gold_candidates = [
        mcif_root / f"en-de_{mcif_set}_refs.yaml",
        mcif_root / "audio-segments.yaml",
    ]
    gold_yaml = next((path for path in gold_candidates if path.is_file()), None)
    if gold_yaml is None:
        raise SystemExit("Could not find MCIF gold audio segmentation YAML")

    entries = _load_yaml_entries(gold_yaml)
    seg_counts = _seg_counts_from_audio_yaml(entries, file_order)
    expected_segments = sum(seg_counts.values())
    if len(entries) != expected_segments:
        raise ValueError(
            f"{gold_yaml} has {len(entries)} segments, expected {expected_segments}"
        )

    audio_yaml = scoring_dir / "audio_definition.yaml"
    audio_yaml.write_text(gold_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"copied gold segments from {gold_yaml}")
    print(f"Wrote {audio_yaml}")

    timing_meta = scoring_dir / "audio_definition.meta.json"
    timing_meta.write_text(
        json.dumps({"timing_source": "gold", "mcif_set": mcif_set}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {timing_meta}")

    directions = {
        "en-de": (mcif_root / f"en-de_{mcif_set}_refs.de", mcif_root / f"en-de_{mcif_set}_refs.en"),
        "en-it": (mcif_root / f"en-it_{mcif_set}_refs.it", mcif_root / f"en-it_{mcif_set}_refs.en"),
    }
    for tag, (ref_path, src_path) in directions.items():
        ref_path = _resolve(ref_path)
        src_path = _resolve(src_path)
        ref_out = scoring_dir / f"refs_{tag}"
        src_out = scoring_dir / f"transcripts_{tag}"
        for out_dir in (ref_out, src_out):
            if out_dir.exists():
                for old in out_dir.glob("*.txt"):
                    old.unlink()

        ref_files = split_reference_file(ref_path, file_order, seg_counts, ref_out, ".txt")
        src_files = split_reference_file(src_path, file_order, seg_counts, src_out, ".txt")
        merged_ref = scoring_dir / f"refs_{tag}_merged.txt"
        merged_src = scoring_dir / f"sources_{tag}_merged.txt"
        merged_ref.write_text(ref_path.read_text(encoding="utf-8"), encoding="utf-8")
        merged_src.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(
            f"{tag}: {len(ref_files)} per-talk refs, merged refs/sources for OmniSTEval "
            f"({merged_ref.name}, {merged_src.name})"
        )


if __name__ == "__main__":
    main()
