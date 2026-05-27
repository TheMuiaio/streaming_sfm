#!/usr/bin/env bash
# Score SimulStream metrics logs with OmniSTEval (IWSLT-style longform resegmentation).
#
# Produces corpus metrics (BLEU, chrF, LongYAAL, …), per-segment instances.resegmented.jsonl,
# and an HTML phrase-level diff report.
#
# Usage:
#   ./score_acl6060_metrics.sh
#   ./score_acl6060_metrics.sh en-de en-pt
#
# Env:
#   OUTPUT_DIR        Directory with metrics_en-*.jsonl (default: output/simulstream_acl6060)
#   ACL6060_ROOT      Dataset cache (default: ~/.cache/simuleval/acl_6060)
#   SPEECH_CFG        speech_processor.yaml (SimulStream eval config for LogReader)
#   PYTHON            Python interpreter
#   BLEU_TOKENIZER    SacreBLEU tokenizer (default: intl)
#   SKIP_COMET=1      Skip COMET (no GPU / model download)
#   HTML_MAX_SEGS=0   Limit segments in HTML report (0 = all)
#
# Requires: pip install 'OmniSTEval[simulstream]'   # optional: OmniSTEval[comet]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/simulstream_acl6060}"
ACL6060_ROOT="${ACL6060_ROOT:-${HOME}/.cache/simuleval/acl_6060}"
SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor.yaml}"
PYTHON="${PYTHON:-python3}"
SCORING_DIR="${OUTPUT_DIR}/scoring_data"
RESULTS_TSV="${OUTPUT_DIR}/scores.tsv"
BLEU_TOKENIZER="${BLEU_TOKENIZER:-intl}"
HTML_MAX_SEGS="${HTML_MAX_SEGS:-0}"

declare -A TARGET_LANG=(
  [en-de]=de
  [en-fr]=fr
  [en-pt]=pt
)

if [[ $# -gt 0 ]]; then
  DIRECTIONS=("$@")
else
  DIRECTIONS=(en-de en-fr en-pt)
fi

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
  echo "  optional COMET: pip install 'OmniSTEval[comet]'" >&2
  exit 1
fi

if ! "$PYTHON" -c "import simulstream" 2>/dev/null; then
  echo "error: simulstream is required to read metrics logs (OmniSTEval[simulstream])." >&2
  exit 1
fi

echo "Preparing ACL 60-60 references, sources, and speech segmentation..."
"$PYTHON" "${REPO_ROOT}/scripts/prepare_acl6060_scoring.py" \
  --acl-root "$ACL6060_ROOT" \
  --output-dir "$OUTPUT_DIR"

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
  echo "========== ${tag} (OmniSTEval longform, lang=${lang}) =========="

  local omnisteval_cmd=("$PYTHON" -m omnisteval.cli)
  if command -v omnisteval >/dev/null 2>&1; then
    omnisteval_cmd=(omnisteval)
  fi

  local -a cmd=(
    "${omnisteval_cmd[@]}" longform
    --speech_segmentation "$SEG_YAML"
    --ref_sentences_file "$ref_merged"
    --hypothesis_file "$metrics_log"
    --hypothesis_format simulstream
    --simulstream_config_file "$SPEECH_CFG"
    --lang "$lang"
    --bleu_tokenizer "$BLEU_TOKENIZER"
    --word_level
    --output_folder "$out_dir"
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

  local instances="${out_dir}/instances.resegmented.jsonl"
  if [[ -f "$instances" ]]; then
    "$PYTHON" "${REPO_ROOT}/scripts/build_omnisteval_html_report.py" \
      --instances "$instances" \
      --output "${out_dir}/phrase_report.html" \
      --title "ACL 60-60 ${tag} — phrase-level errors" \
      --max-instances "$HTML_MAX_SEGS"
    echo "Phrase report: ${out_dir}/phrase_report.html"
  fi

  echo "OmniSTEval outputs: ${out_dir}/"
}

for tag in "${DIRECTIONS[@]}"; do
  score_direction "$tag"
done

echo ""
echo "Wrote summary: $RESULTS_TSV"
column -t -s $'\t' "$RESULTS_TSV" 2>/dev/null || cat "$RESULTS_TSV"
