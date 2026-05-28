#!/usr/bin/env bash
# Run SimulStream offline inference (python -m simulstream.inference) with
# agent_simulstream.CascadeSpeechProcessor on ACL 60-60 eval audio for en-de and en-fr.
#
# Usage:
#   ./run_acl6060_simulstream.sh
# Optional env:
#   ACL6060_ROOT   Dataset root (default: ~/.cache/simuleval/acl_6060)
#   SPEECH_CFG     speech_processor YAML (default: repo/speech_processor.yaml)
#   OUTPUT_DIR     Where metrics logs and wav list link farm go
#   PYTHON         Python interpreter (default: python3; auto-falls back to conda envs with mweralign)
#   SKIP_COMET=1   Skip COMET scoring
#
# After inference, scores with OmniSTEval (BLEU, chrF, LongYAAL, phrase HTML report).
# Requires: pip install 'OmniSTEval[simulstream]' simulstream
#   optional COMET: pip install unbabel-comet  (or pip install 'OmniSTEval[comet]')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

ACL6060_ROOT="${ACL6060_ROOT:-${HOME}/.cache/simuleval/acl_6060}"
#SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor_qwen35_9b.yaml}"
SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/simulstream_acl6060}"
PYTHON="${PYTHON:-python3}"

# SimulStream's wav list loader resolves paths as: dirname(list_file) + "/" + line
# The cache FILE_ORDER lists utterance IDs without extension; wavs live in eval/full_wavs/.
LINKFARM="${OUTPUT_DIR}/eval_wav_linkfarm"
FILE_ORDER="${ACL6060_ROOT}/en-de_eval_wavs_list.txt"

if [[ ! -f "$FILE_ORDER" ]]; then
  echo "error: missing wav list / FILE_ORDER at: $FILE_ORDER" >&2
  echo "Set ACL6060_ROOT to your acl_6060 cache directory." >&2
  exit 1
fi

if [[ ! -f "$SPEECH_CFG" ]]; then
  echo "error: speech processor config not found: $SPEECH_CFG" >&2
  exit 1
fi

EVAL_DIR="$(cd "$(dirname "$(readlink -f "$FILE_ORDER" 2>/dev/null || realpath "$FILE_ORDER")")" && pwd)"
FULL_WAVS="${EVAL_DIR}/full_wavs"

if [[ ! -d "$FULL_WAVS" ]]; then
  echo "error: expected wav directory not found: $FULL_WAVS" >&2
  exit 1
fi

mkdir -p "$LINKFARM"
WAV_LIST_FILE="${LINKFARM}/wav_list.txt"
: >"$WAV_LIST_FILE"

while IFS= read -r id || [[ -n "$id" ]]; do
  [[ -z "${id// }" ]] && continue
  wav="${FULL_WAVS}/${id}.wav"
  if [[ ! -f "$wav" ]]; then
    echo "error: missing wav for id '$id': $wav" >&2
    exit 1
  fi
  ln -sf "$wav" "${LINKFARM}/${id}.wav"
  echo "${id}.wav" >>"$WAV_LIST_FILE"
done <"$FILE_ORDER"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_direction() {
  local tag="$1"
  local src_lang="$2"
  local tgt_lang="$3"
  local metrics="${OUTPUT_DIR}/metrics_${tag}.jsonl"

  echo "=== ${tag} (${src_lang} -> ${tgt_lang}) ==="
  "$PYTHON" -m simulstream.inference \
    --speech-processor-config "$SPEECH_CFG" \
    --wav-list-file "$WAV_LIST_FILE" \
    --src-lang "$src_lang" \
    --tgt-lang "$tgt_lang" \
    --metrics-log-file "$metrics"
  echo "Wrote metrics: $metrics"
}

mkdir -p "$OUTPUT_DIR"

run_direction "en-de" "English" "German"
run_direction "en-fr" "English" "French"
run_direction "en-pt" "English" "Portuguese"

echo ""
echo "Scoring with OmniSTEval (BLEU, chrF, LongYAAL, phrase report)..."
"${REPO_ROOT}/score_acl6060_metrics.sh" en-de
"${REPO_ROOT}/score_acl6060_metrics.sh" en-fr
"${REPO_ROOT}/score_acl6060_metrics.sh" en-pt

echo "Done. Wav list (for SimulStream): $WAV_LIST_FILE"
