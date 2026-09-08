"""Resolve eval wav lists / chunk counts for mid-run progress logging."""

from __future__ import annotations

import logging
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ACL_ROOT = Path.home() / ".cache" / "simuleval" / "acl_6060"
DEFAULT_MCIF_ROOT = Path.home() / ".cache" / "simuleval" / "mcif_iwslt26"


@dataclass(frozen=True)
class AudioProgressEntry:
    audio_id: str
    wav_path: Path
    num_chunks: int


@dataclass(frozen=True)
class EvalProgressIndex:
    dataset: str
    split: str
    list_path: Path
    entries: Tuple[AudioProgressEntry, ...]

    def get(self, speech_id: int) -> Optional[AudioProgressEntry]:
        if 0 <= speech_id < len(self.entries):
            return self.entries[speech_id]
        return None


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    return default


def _config_get(config: Optional[SimpleNamespace], *names: str):
    if config is None:
        return None
    for name in names:
        value = getattr(config, name, None)
        if value is not None and str(value).strip() != "":
            return value
    return None


def resolve_dataset_and_split(
    config: Optional[SimpleNamespace] = None,
) -> Tuple[str, str]:
    """
    Decide which eval corpus + split to use.

    Preference order:
      1. Explicit config ``eval_dataset`` / ``eval_split`` (aliases: dataset, acl_set, mcif_set)
      2. ``OUTPUT_DIR`` hint (``mcif`` in path => MCIF)
      3. Whichever of ``MCIF_SET`` / ``ACL6060_SET`` is present in the environment
      4. Default: ACL ``eval``
    """
    dataset = _config_get(config, "eval_dataset", "dataset")
    split = _config_get(config, "eval_split", "acl_set", "mcif_set")

    mcif_set = os.environ.get("MCIF_SET")
    acl_set = os.environ.get("ACL6060_SET")
    output_dir = (os.environ.get("OUTPUT_DIR") or "").lower()

    if dataset is not None:
        dataset_l = str(dataset).strip().lower()
        if dataset_l in {"acl", "acl6060", "acl_6060"}:
            return "acl", str(split or acl_set or "eval")
        if dataset_l in {"mcif", "mcif_iwslt26"}:
            return "mcif", str(split or mcif_set or "test")
        raise ValueError(f"Unsupported eval_dataset={dataset!r}")

    if "mcif" in output_dir:
        return "mcif", str(split or mcif_set or "test")

    if mcif_set and not acl_set:
        return "mcif", str(split or mcif_set)
    if acl_set and not mcif_set:
        return "acl", str(split or acl_set)
    if mcif_set and acl_set:
        # Both leftover in the parent env: prefer ACL unless OUTPUT_DIR already hinted MCIF.
        return "acl", str(split or acl_set)

    if split is not None:
        split_s = str(split)
        if split_s in {"test"} or split_s.startswith("test"):
            return "mcif", split_s
        return "acl", split_s

    return "acl", "eval"


def resolve_wav_list_path(
    dataset: str,
    split: str,
    config: Optional[SimpleNamespace] = None,
) -> Path:
    explicit = _config_get(config, "eval_wav_list", "wav_list_file")
    if explicit is not None:
        return Path(str(explicit)).expanduser().resolve()

    if dataset == "acl":
        root = _env_path("ACL6060_ROOT", DEFAULT_ACL_ROOT)
        # Same entry point as run_acl6060_simulstream.sh (symlink to .../{split}/FILE_ORDER).
        return (root / f"en-de_{split}_wavs_list.txt").resolve()

    if dataset == "mcif":
        root = _env_path("MCIF_ROOT", DEFAULT_MCIF_ROOT)
        return (root / f"en-de_{split}_wavs_list.txt").resolve()

    raise ValueError(f"Unsupported dataset={dataset!r}")


def _read_list_lines(list_path: Path) -> List[str]:
    lines: List[str] = []
    with list_path.open(encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                lines.append(item)
    return lines


def _wav_num_chunks(wav_path: Path, speech_chunk_size: float) -> int:
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
    samples_per_chunk = int(sample_rate * float(speech_chunk_size))
    if samples_per_chunk <= 0:
        return 0
    return max(1, math.ceil(n_frames / samples_per_chunk))


def _resolve_acl_wav(list_path: Path, utt_id: str) -> Path:
    # list_path may be the top-level symlink or the real .../{split}/FILE_ORDER.
    split_dir = list_path.resolve().parent
    candidate = split_dir / "full_wavs" / f"{utt_id}.wav"
    if candidate.is_file():
        return candidate
    # Fallback: id already includes .wav, or wavs next to the list.
    for name in (utt_id, f"{utt_id}.wav"):
        alt = split_dir / name
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"ACL wav not found for id={utt_id!r} under {split_dir}")


def _resolve_mcif_wav(list_path: Path, rel_path: str) -> Path:
    root = list_path.resolve().parent
    candidate = (root / rel_path).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MCIF wav not found for path={rel_path!r} under {root}")


def build_eval_progress_index(
    speech_chunk_size: float,
    config: Optional[SimpleNamespace] = None,
) -> Optional[EvalProgressIndex]:
    """
    Preload audio ids + expected chunk counts from the active eval list.

    Returns None (and logs a warning) if the list / wavs cannot be resolved.
    """
    try:
        dataset, split = resolve_dataset_and_split(config)
        list_path = resolve_wav_list_path(dataset, split, config)
        if not list_path.is_file():
            logger.warning(
                "Eval progress: wav list not found for dataset=%s split=%s path=%s",
                dataset,
                split,
                list_path,
            )
            return None

        lines = _read_list_lines(list_path)
        entries: List[AudioProgressEntry] = []
        for line in lines:
            if dataset == "acl":
                utt_id = line[:-4] if line.endswith(".wav") else line
                wav_path = _resolve_acl_wav(list_path, utt_id)
                audio_id = utt_id
            else:
                wav_path = _resolve_mcif_wav(list_path, line)
                audio_id = Path(line).stem
            num_chunks = _wav_num_chunks(wav_path, speech_chunk_size)
            entries.append(
                AudioProgressEntry(
                    audio_id=audio_id,
                    wav_path=wav_path,
                    num_chunks=num_chunks,
                )
            )

        index = EvalProgressIndex(
            dataset=dataset,
            split=split,
            list_path=list_path,
            entries=tuple(entries),
        )
        logger.info(
            "Eval progress index ready: dataset=%s split=%s list=%s audios=%d",
            dataset,
            split,
            list_path,
            len(entries),
        )
        return index
    except Exception as exc:  # noqa: BLE001 - progress logging must never break inference
        logger.warning("Eval progress: failed to build index (%s)", exc)
        return None


def format_step_banner(
    index: Optional[EvalProgressIndex],
    speech_id: int,
    step: int,
) -> str:
    entry = index.get(speech_id) if index is not None else None
    total = entry.num_chunks if entry is not None else "?"
    return (
        f"audio_id {speech_id} | "
        f"================ Performing new step {step} / {total} ================"
    )
