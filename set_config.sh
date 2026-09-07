#!/usr/bin/env bash
# Shared configuration for ACL 60-60 SimulStream inference and scoring.
#
# Sourced by run_acl6060_simulstream.sh and score_acl6060_metrics.sh.
# Edit the defaults below, or override any variable when invoking the scripts:
#   SPEECH_CFG=/path/to/other.yaml OUTPUT_DIR=/tmp/my_run ./run_acl6060_simulstream.sh
#
# Main knobs:
#   SPEECH_CFG     speech_processor YAML (drives model / pipeline settings)
#   OUTPUT_DIR     metrics logs, scoring data, OmniSTEval outputs (default: output/<config-basename>)
#   ACL6060_ROOT   ACL 60-60 dataset cache
#   ACL6060_SET    ACL 60-60 split: eval (default) or dev
#   PYTHON         Python interpreter
#   BLEU_TOKENIZER SacreBLEU tokenizer for OmniSTEval (default: intl)
#   HTML_MAX_SEGS  Max segments in phrase HTML report (0 = all)
#   SKIP_COMET=1   Skip COMET scoring
#   COMET_MODEL    COMET checkpoint on Hugging Face (default: Unbabel/XCOMET-XL)

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$_CONFIG_DIR"
REPO_ROOT="$SCRIPT_DIR"

ACL6060_ROOT="${ACL6060_ROOT:-${HOME}/.cache/simuleval/acl_6060}"
ACL6060_SET="${ACL6060_SET:-eval}"
export ACL6060_ROOT
export ACL6060_SET
if [[ "$ACL6060_SET" != eval && "$ACL6060_SET" != dev ]]; then
  echo "error: ACL6060_SET must be 'eval' or 'dev' (got: $ACL6060_SET)" >&2
  exit 1
fi

# Default speech processor config (change this line to switch experiments).
#SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor_qwen35_27b_bnb4_spec.yaml}"
SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor_policies.yaml}"
#SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor_qwen35_9b_bnb4_smallerW.yaml}"
#SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor.yaml}"

speechp_name="$(basename "${SPEECH_CFG}" .yaml)"
#OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/simulstream_acl6060}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  _OUTPUT_DIR_WAS_SET=1
else
  _OUTPUT_DIR_WAS_SET=0
fi

acl6060_default_output_dir() {
  local dir="${REPO_ROOT}/output/${speechp_name}"
  if [[ "$ACL6060_SET" != eval ]]; then
    dir="${dir}_${ACL6060_SET}"
  fi
  printf '%s' "$dir"
}

acl6060_refresh_output_paths() {
  if [[ $_OUTPUT_DIR_WAS_SET -eq 0 ]]; then
    OUTPUT_DIR="$(acl6060_default_output_dir)"
  fi
  SCORING_DIR="${OUTPUT_DIR}/scoring_data"
  RESULTS_TSV="${OUTPUT_DIR}/scores.tsv"
  PREDICTS="${OUTPUT_DIR}/preds.txt"
}

OUTPUT_DIR="${OUTPUT_DIR:-$(acl6060_default_output_dir)}"

PYTHON="${PYTHON:-python3}"
BLEU_TOKENIZER="${BLEU_TOKENIZER:-13a}"
HTML_MAX_SEGS="${HTML_MAX_SEGS:-0}"
LATENCY_UNIT="${LATENCY_UNIT:-word}"
COMET_MODEL="${COMET_MODEL:-Unbabel/XCOMET-XL}"

acl6060_refresh_output_paths

# Parse optional first argument as ACL 60-60 split (eval|dev); remaining args are directions.
# Usage: acl6060_parse_split_and_directions DIRECTIONS_ARRAY_NAME "$@"
acl6060_parse_split_and_directions() {
  local -n _directions=$1
  shift
  _directions=(en-de en-fr en-nl en-pt en-ru en-tr)
  if [[ $# -eq 0 ]]; then
    acl6060_refresh_output_paths
    return 0
  fi
  if [[ $1 == eval || $1 == dev ]]; then
    ACL6060_SET="$1"
    shift
  fi
  if [[ $# -gt 0 ]]; then
    _directions=("$@")
  fi
  export ACL6060_SET
  acl6060_refresh_output_paths
}
