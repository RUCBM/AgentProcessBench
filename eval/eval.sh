#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#    bash eval/eval.sh \
#      --model gpt-5.2 \
#      --base_url "https://your-openai-compatible-endpoint/v1" \
#      --api_key "your_api_key" \
#      --datasets "bfcl,hotpotqa,tau2,gaia_dev" \
#      --concurrency 8

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/AgentProcessBench"
RESULTS_ROOT="$PROJECT_ROOT/eval/yourresults"
DATASETS="bfcl,hotpotqa,tau2,gaia_dev"
MODEL=""
RUN_NAME=""
BASE_URL="${OPENAI_BASE_URL:-}"
API_KEY="${OPENAI_API_KEY:-}"

CONCURRENCY=8
TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKENS=8192
TIMEOUT_S=300
START=0
END=-1

usage() {
  cat <<EOF
Usage:
  bash eval.sh --model <model_name> [options]

Required:
  --model <name>                  OpenAI-compatible model name

Options:
  --base_url <url>                Override OPENAI_BASE_URL
  --api_key <key>                 Override OPENAI_API_KEY
  --datasets <csv>                Default: $DATASETS
  --data_dir <dir>                Default: $DATA_DIR
  --results_root <dir>            Default: $RESULTS_ROOT
  --run_name <name>               Default: sanitized model name
  --concurrency <int>             Default: $CONCURRENCY
  --temperature <float>           Default: $TEMPERATURE
  --top_p <float>                 Default: $TOP_P
  --max_tokens <int>              Default: $MAX_TOKENS
  --timeout_s <int>               Default: $TIMEOUT_S
  --start <int>                   Default: $START
  --end <int>                     Default: $END
  -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base_url)
      BASE_URL="$2"
      shift 2
      ;;
    --api_key)
      API_KEY="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --timeout_s)
      TIMEOUT_S="$2"
      shift 2
      ;;
    --start)
      START="$2"
      shift 2
      ;;
    --end)
      END="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Missing required arg: --model" >&2
  usage
  exit 2
fi

if [[ -z "$BASE_URL" ]]; then
  echo "Missing base URL. Pass --base_url or set OPENAI_BASE_URL." >&2
  exit 2
fi

if [[ -z "$API_KEY" ]]; then
  echo "Missing API key. Pass --api_key or set OPENAI_API_KEY." >&2
  exit 2
fi

if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --concurrency: $CONCURRENCY (must be a positive integer)." >&2
  exit 2
fi

if ! python -c "import openai" >/dev/null 2>&1; then
  echo "Python package 'openai' is not installed in current environment." >&2
  echo "Install it first, e.g.: pip install openai" >&2
  exit 2
fi

MODEL_TAG="$(echo "$MODEL" | sed 's#[ /]#_#g')"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$MODEL_TAG"
fi

if [[ "$MODEL_TAG" == *deepseek* || "$BASE_URL" == *deepseek* ]]; then
  if [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] && (( MAX_TOKENS > 8192 )); then
    echo "[eval] DeepSeek detected, clamp max_tokens from $MAX_TOKENS to 8192"
    MAX_TOKENS=8192
  fi
fi

RUN_DIR="$RESULTS_ROOT/$RUN_NAME"
RAW_DIR="$RESULTS_ROOT/_raw/$RUN_NAME"
mkdir -p "$RUN_DIR" "$RAW_DIR"

echo "[eval] model=$MODEL datasets=$DATASETS concurrency=$CONCURRENCY"

IFS=',' read -r -a DS_ARR <<< "$DATASETS"
DATASETS_NORMALIZED=()
for ds in "${DS_ARR[@]}"; do
  ds="${ds//[[:space:]]/}"
  [[ -z "$ds" ]] && continue
  DATASETS_NORMALIZED+=("$ds")
done

if [[ ${#DATASETS_NORMALIZED[@]} -eq 0 ]]; then
  echo "No valid datasets in --datasets" >&2
  exit 2
fi

for ds in "${DATASETS_NORMALIZED[@]}"; do
  INPUT_PATH="$DATA_DIR/${ds}.jsonl"
  if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Dataset file not found: $INPUT_PATH" >&2
    exit 2
  fi

  OUTPUT_PATH="$RUN_DIR/${ds}__blind_${MODEL_TAG}.jsonl"

  CMD=(
    python "$PROJECT_ROOT/eval/llm_annotation.py"
    --input_path "$INPUT_PATH"
    --dataset "$ds"
    --model "$MODEL"
    --base_url "$BASE_URL"
    --api_key "$API_KEY"
    --output_path "$OUTPUT_PATH"
    --raw_output_dir "$RAW_DIR"
    --concurrency "$CONCURRENCY"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_tokens "$MAX_TOKENS"
    --timeout_s "$TIMEOUT_S"
    --start "$START"
    --end "$END"
  )

  echo "[eval] annotate dataset=$ds -> $OUTPUT_PATH"
  "${CMD[@]}"
done

COMPARE_LOG="$RUN_DIR/score.txt"
DATASETS_JOINED="$(IFS=','; echo "${DATASETS_NORMALIZED[*]}")"

echo "[eval] compare predictions vs reference"
python "$PROJECT_ROOT/eval/compare.py" \
  --reference_dir "$DATA_DIR" \
  --models_root_dir "$RUN_DIR" \
  --datasets "$DATASETS_JOINED" \
  --run_name_grouping raw | tee "$COMPARE_LOG"

AVG_LINE="$(awk '$1=="AVG"{line=$0} END{print line}' "$COMPARE_LOG")"
if [[ -z "$AVG_LINE" ]]; then
  echo "[eval] AVG row not found in $COMPARE_LOG" >&2
  exit 1
fi

PARSE_FAILED=0
for ds in "${DATASETS_NORMALIZED[@]}"; do
  DS_LINE="$(awk -v ds="$ds" '$1==ds{line=$0} END{print line}' "$COMPARE_LOG")"
  if [[ -z "$DS_LINE" ]]; then
    echo "[eval] Missing dataset row in score table: $ds" >&2
    PARSE_FAILED=1
    continue
  fi

  DS_STEP_MICRO_ACC="$(awk -v ds="$ds" '$1==ds{val=$5} END{print val}' "$COMPARE_LOG")"
  DS_FIRSTERROR_ACC="$(awk -v ds="$ds" '$1==ds{val=$7} END{print val}' "$COMPARE_LOG")"
  if [[ -z "$DS_STEP_MICRO_ACC" || -z "$DS_FIRSTERROR_ACC" ]]; then
    echo "[eval] Failed to parse metrics for dataset=$ds from row: $DS_LINE" >&2
    PARSE_FAILED=1
    continue
  fi

  echo "[eval] dataset=$ds step_micro_acc=$DS_STEP_MICRO_ACC firsterroracc=$DS_FIRSTERROR_ACC"
done

STEP_MICRO_ACC="$(awk '$1=="AVG"{val=$5} END{print val}' "$COMPARE_LOG")"
FIRSTERROR_ACC="$(awk '$1=="AVG"{val=$7} END{print val}' "$COMPARE_LOG")"

if [[ -z "$STEP_MICRO_ACC" || -z "$FIRSTERROR_ACC" ]]; then
  echo "[eval] Failed to parse step_micro_acc/firsterroracc from AVG row: $AVG_LINE" >&2
  exit 1
fi

if [[ "$PARSE_FAILED" -ne 0 ]]; then
  exit 1
fi

echo "[eval] overall step_micro_acc=$STEP_MICRO_ACC"
echo "[eval] overall firsterroracc=$FIRSTERROR_ACC"
