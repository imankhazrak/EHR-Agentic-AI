#!/usr/bin/env bash
# Run strict-v2 export validation + tokenization + label-mask audits (CPU-friendly).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

DATA_DIR="$PROJECT_DIR/dataset_for_unsloth_multitask_strict_v2"
AUDIT_DIR="$PROJECT_DIR/outputs/audits"

echo "=== Export strict_v2 JSONL ==="
python -m src.scripts.export_multitask_unsloth_jsonl \
  --instruction-version strict_v2 \
  --compact-output-json \
  --reasoning-style minimal \
  --out-dir "$DATA_DIR"

echo "=== Tokenization audit (head 5 train) ==="
mkdir -p "$AUDIT_DIR"
python scripts/audit_unsloth_multitask_tokenization.py \
  --jsonl "$DATA_DIR/train.jsonl" \
  --model-name unsloth/gemma-4-e2b-it-unsloth-bnb-4bit \
  --max-seq-length 2048 \
  --num-samples 5 \
  --sample-mode head \
  --tokenizer-only \
  --output-md "$AUDIT_DIR/unsloth_tokenization_audit_strict_v2_train_head5.md" \
  --fail-on-error

echo "=== Label-mask audit (5 train rows) ==="
python scripts/train_unsloth_router.py \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --train-max-samples 5 \
  --debug-only \
  --debug-label-mask-check 5 \
  --debug-label-mask-md "$AUDIT_DIR/label_mask_audit_strict_v2_head5.md" \
  --tokenizer-only

echo "=== Pre-smoke audits PASS ==="
echo "Train smoke: sbatch slurm/train_unsloth_strict_v2_smoke.slurm"
