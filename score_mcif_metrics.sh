#!/usr/bin/env bash
# Score SimulStream metrics logs for MCIF with OmniSTEval longform resegmentation.
#
# Usage:
#   ./score_mcif_metrics.sh
#   ./score_mcif_metrics.sh en-de en-it

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_mcif_config.sh"

declare -A TARGET_LANG=(
  [en-de]=de
  [en-it]=it
)

declare -a DIRECTIONS
mcif_parse_directions DIRECTIONS "$@"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

find_python_with_omnisteval() {
  if "$PYTHON" -c "import omnisteval" 2>/dev/null; then
    return 0
  fi
  local candidate
  for candidate in \
    "${HOME}/miniconda3/envs/iwslt/bin/python" \
    "${HOME}/miniconda3/envs/evaluation/bin/python" \
    "${HOME}/miniconda3/envs/simuleval/bin/python"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import omnisteval" 2>/dev/null; then
      PYTHON="$candidate"
      echo "Using PYTHON=$PYTHON (has omnisteval)"
      return 0
    fi
  done
  return 1
}

if ! find_python_with_omnisteval; then
  echo "error: OmniSTEval is required for scoring." >&2
  echo "Install with: pip install 'OmniSTEval[simulstream]'" >&2
  exit 1
fi

if ! "$PYTHON" -c "import simulstream" 2>/dev/null; then
  echo "error: simulstream is required to read metrics logs." >&2
  exit 1
fi

echo "Preparing MCIF references, sources, and speech segmentation (set=${MCIF_SET})..."
"$PYTHON" "${REPO_ROOT}/scripts/prepare_mcif_scoring.py" \
  --mcif-root "$MCIF_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --set "$MCIF_SET"

SEG_YAML="${SCORING_DIR}/audio_definition.yaml"
if [[ ! -f "$SEG_YAML" ]]; then
  echo "error: missing $SEG_YAML" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
: >"$RESULTS_TSV"
printf "direction\tmetric\tvalue\tdetails\n" >>"$RESULTS_TSV"

append_scores_from_tsv() {
  local tag="$1"
  local scores_tsv="$2"
  [[ -f "$scores_tsv" ]] || return 0
  tail -n +2 "$scores_tsv" | while IFS=$'\t' read -r metric value _; do
    printf "%s\t%s\t%s\tOmniSTEval longform\n" "$tag" "$metric" "$value" >>"$RESULTS_TSV"
  done
}

append_simulstream_stats() {
  local tag="$1"
  local metrics_log="$2"
  local out_dir="$3"
  local stats_json="${out_dir}/stats.json"

  echo "Computing SimulStream stats from ${metrics_log}..."
  "$PYTHON" - <<'PY' "$SPEECH_CFG" "$metrics_log" "$LATENCY_UNIT" "$stats_json" "$RESULTS_TSV" "$tag"
import json
import statistics
import sys
from pathlib import Path

from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader
from simulstream.metrics.stats import NormalizedErasure, RealTimeFactor

eval_config_path, metrics_log, latency_unit, stats_path, results_tsv, tag = sys.argv[1:7]
eval_config = yaml_config(eval_config_path)
log_reader = LogReader(eval_config, metrics_log, latency_unit=latency_unit)
stats = {
    stat.name(): {"description": stat.description(), "value": stat.compute(log_reader)}
    for stat in (NormalizedErasure(), RealTimeFactor())
}
rows = []
with open(metrics_log, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "total_audio_processed" in row:
            rows.append(row)

generated_counts = [len(row.get("generated_tokens", [])) for row in rows]
deleted_counts = [len(row.get("deleted_tokens", [])) for row in rows]
compute_times = [float(row.get("computation_time", 0.0) or 0.0) for row in rows]
generated_total = sum(generated_counts)
deleted_total = sum(deleted_counts)
emitting_steps = sum(1 for n in generated_counts if n)
revision_steps = sum(1 for n in deleted_counts if n)
max_audio = max((float(row.get("total_audio_processed", 0.0) or 0.0) for row in rows), default=0.0)

def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]

extra_stats = {
    "chunks": {"description": "Number of streaming chunks in the metrics log.", "value": len(rows)},
    "generated_tokens_total": {
        "description": "Total generated target tokens, including tokens later deleted.",
        "value": generated_total,
    },
    "deleted_tokens_total": {"description": "Total deleted target tokens.", "value": deleted_total},
    "deletion_ratio": {
        "description": "Deleted target tokens divided by generated target tokens.",
        "value": deleted_total / generated_total if generated_total else 0.0,
    },
    "revision_step_rate": {
        "description": "Fraction of chunks that delete at least one target token.",
        "value": revision_steps / len(rows) if rows else 0.0,
    },
    "emission_step_rate": {
        "description": "Fraction of chunks that emit at least one target token.",
        "value": emitting_steps / len(rows) if rows else 0.0,
    },
    "mean_computation_time_s": {
        "description": "Mean wall-clock computation time per streaming chunk.",
        "value": statistics.fmean(compute_times) if compute_times else 0.0,
    },
    "p95_computation_time_s": {
        "description": "95th percentile wall-clock computation time per streaming chunk.",
        "value": percentile(compute_times, 95),
    },
    "max_computation_time_s": {
        "description": "Maximum wall-clock computation time for a streaming chunk.",
        "value": max(compute_times, default=0.0),
    },
    "total_computation_time_s": {
        "description": "Total wall-clock computation time across streaming chunks.",
        "value": sum(compute_times),
    },
    "metrics_log_rtf": {
        "description": "Total computation time divided by processed audio duration.",
        "value": sum(compute_times) / max_audio if max_audio else 0.0,
    },
}
stats.update(extra_stats)
Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
with open(results_tsv, "a", encoding="utf-8") as f:
    for name, payload in stats.items():
        f.write(f"{tag}\t{name}\t{payload['value']}\tsimulstream.metrics.stats\n")
for name, payload in stats.items():
    print(f"  {name}: {payload['value']:.6f}")
PY
}

score_direction() {
  local tag="$1"
  local lang="${TARGET_LANG[$tag]:-}"
  local metrics_log="${OUTPUT_DIR}/metrics_${tag}.jsonl"
  local ref_merged="${SCORING_DIR}/refs_${tag}_merged.txt"
  local src_merged="${SCORING_DIR}/sources_${tag}_merged.txt"
  local out_dir="${OUTPUT_DIR}/omnisteval_${tag}"

  if [[ -z "$lang" ]]; then
    echo "error: unknown direction '$tag' (supported: ${!TARGET_LANG[*]})" >&2
    return 1
  fi
  if [[ ! -f "$metrics_log" ]]; then
    echo "error: metrics log not found: $metrics_log (run inference first)" >&2
    return 1
  fi
  if [[ ! -f "$ref_merged" ]]; then
    echo "error: merged references not found: $ref_merged" >&2
    return 1
  fi

  echo ""
  echo "========== ${tag} (OmniSTEval longform, set=${MCIF_SET}, lang=${lang}) =========="

  local omnisteval_cmd=("$PYTHON" -m omnisteval.cli)
  if command -v omnisteval >/dev/null 2>&1; then
    omnisteval_cmd=(omnisteval)
  fi

  local -a cmd=(
    "${omnisteval_cmd[@]}" longform
    --speech_segmentation "$SEG_YAML"
    --ref_sentences_file "$ref_merged"
    --hypothesis_file "$metrics_log"
    --source_sentences_file "$src_merged"
    --lang "$lang"
    --bleu_tokenizer "${BLEU_TOKENIZER}"
    --output_folder "$out_dir"
    --hypothesis_format simulstream
    --simulstream_config_file "$SPEECH_CFG"
    --word_level
  )

  if [[ "${SKIP_COMET:-0}" != "1" ]]; then
    if [[ -f "$src_merged" ]] && "$PYTHON" -c "import comet" 2>/dev/null; then
      cmd+=(--comet --source_sentences_file "$src_merged")
    elif [[ ! -f "$src_merged" ]]; then
      echo "Skipping COMET (missing $src_merged)" >&2
    else
      echo "Skipping COMET (install with: pip install unbabel-comet)" >&2
    fi
  else
    echo "Skipping COMET (SKIP_COMET=1)" >&2
  fi

  "${cmd[@]}"

  append_scores_from_tsv "$tag" "${out_dir}/scores.tsv"
  append_simulstream_stats "$tag" "$metrics_log" "$out_dir"

  local instances="${out_dir}/instances.resegmented.jsonl"
  if [[ -f "$instances" ]]; then
    "$PYTHON" "${REPO_ROOT}/scripts/build_omnisteval_html_report.py" \
      --instances "$instances" \
      --output "${out_dir}/phrase_report.html" \
      --title "MCIF ${MCIF_SET} ${tag} — phrase-level errors" \
      --max-instances "$HTML_MAX_SEGS"
    echo "Phrase report: ${out_dir}/phrase_report.html"
  fi

  : >"${out_dir}/preds.txt"
  if [[ -f "$instances" ]]; then
    "$PYTHON" - <<'PY' "$instances" "${out_dir}/preds.txt"
import json
import sys
from collections import defaultdict
from pathlib import Path

instances_path = Path(sys.argv[1])
preds_path = Path(sys.argv[2])
by_doc: dict[int, list[str]] = defaultdict(list)
for line in instances_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    by_doc[int(row["docid"])].append(row["prediction"])
with preds_path.open("w", encoding="utf-8") as f:
    for docid in sorted(by_doc):
        f.write(" ".join(by_doc[docid]).strip() + "\n")
PY
  fi

  echo "OmniSTEval outputs: ${out_dir}/"
}

for tag in "${DIRECTIONS[@]}"; do
  score_direction "$tag"
done

echo ""
echo "Wrote summary: $RESULTS_TSV"
column -t -s $'\t' "$RESULTS_TSV" 2>/dev/null || cat "$RESULTS_TSV"
