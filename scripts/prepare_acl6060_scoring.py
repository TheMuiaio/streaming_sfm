#!/usr/bin/env python3
"""Prepare per-talk reference files and segment timing YAML for SimulStream metrics."""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import yaml


def _resolve(path: Path) -> Path:
    return path.resolve()


def _seg_counts_from_xml(xml_path: Path) -> dict[str, int]:
    text = xml_path.read_text(encoding="utf-8")
    talk_ids = re.findall(r"<talkid>([^<]+)</talkid>", text)
    parts = re.split(r"<talkid>[^<]+</talkid>", text)[1:]
    counts = [len(re.findall(r"<seg id=", part)) for part in parts]
    if len(talk_ids) != len(counts):
        raise RuntimeError(f"Could not parse segment counts from {xml_path}")
    return dict(zip(talk_ids, counts))


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _load_yaml_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {path}")
    return data


def _try_existing_audio_definition(candidates: list[Path], expected_segments: int) -> Path | None:
    for candidate in candidates:
        candidate = _resolve(candidate)
        if not candidate.is_file():
            continue
        try:
            entries = _load_yaml_entries(candidate)
        except Exception:
            continue
        if len(entries) == expected_segments:
            return candidate
    return None


def build_audio_definition(
    acl_root: Path,
    eval_dir: Path,
    file_order: list[str],
    seg_counts: dict[str, int],
    out_yaml: Path,
) -> str:
    expected_segments = sum(seg_counts[talk] for talk in file_order)
    candidates = [
        acl_root / "en-de_eval_refs.yaml",
        acl_root / "en-fr_eval_refs.yaml",
    ]
    existing = _try_existing_audio_definition(candidates, expected_segments)
    if existing is not None:
        out_yaml.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
        return f"copied gold segments from {existing}"

    full_wavs = eval_dir / "full_wavs"
    entries: list[dict] = []
    for talk in file_order:
        wav_name = f"{talk}.wav"
        duration = _wav_duration_seconds(full_wavs / wav_name)
        n_segs = seg_counts[talk]
        seg_dur = duration / n_segs
        offset = 0.0
        for _ in range(n_segs):
            entries.append(
                {
                    "wav": wav_name,
                    "offset": round(offset, 4),
                    "duration": round(seg_dur, 4),
                }
            )
            offset += seg_dur

    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False)
    return (
        f"built proportional segment timings ({expected_segments} segments) "
        "(set ACL6060_ROOT/.../en-de_eval_refs.yaml for gold timings)"
    )


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
            f"{ref_path} has {len(lines)} lines, expected {expected} from XML segment counts"
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
        "--acl-root",
        type=Path,
        default=Path.home() / ".cache/simuleval/acl_6060",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    acl_root = _resolve(args.acl_root)
    output_dir = _resolve(args.output_dir)
    scoring_dir = output_dir / "scoring_data"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    file_order_path = acl_root / "en-de_eval_wavs_list.txt"
    file_order_path = _resolve(file_order_path)
    eval_dir = file_order_path.parent
    file_order = [
        line.strip()
        for line in file_order_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    xml_path = eval_dir / "text/xml/ACL.6060.eval.en-xx.de.xml"
    seg_counts = _seg_counts_from_xml(xml_path)

    audio_yaml = scoring_dir / "audio_definition.yaml"
    note = build_audio_definition(acl_root, eval_dir, file_order, seg_counts, audio_yaml)
    print(note)
    print(f"Wrote {audio_yaml}")

    directions = {
        "en-de": (acl_root / "en-de_eval_refs.de", acl_root / "en-de_eval_refs.en"),
        "en-fr": (acl_root / "en-fr_eval_refs.fr", acl_root / "en-fr_eval_refs.en"),
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
        # File stems must match wav stems (e.g. 2022.acl-long.410) for SimulStream LogReader.
        ref_files = split_reference_file(
            ref_path,
            file_order,
            seg_counts,
            ref_out,
            ".txt",
        )
        src_files = split_reference_file(
            src_path,
            file_order,
            seg_counts,
            src_out,
            ".txt",
        )
        print(f"{tag}: {len(ref_files)} reference files under {scoring_dir / f'refs_{tag}'}")


if __name__ == "__main__":
    main()
