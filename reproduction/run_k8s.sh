#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="/shared/rarg-2607-24223"
ASSET_ROOT="$SHARED_ROOT/assets"
HF_CACHE="$SHARED_ROOT/hf-cache"
CORPUS_ROOT="$ASSET_ROOT/bc_plus_100k"
PARQUET_PATH="$ASSET_ROOT/browsecomp_plus/data.parquet"
INDEX_ROOT="$ASSET_ROOT/index-qwen3-embedding-0.6b"
AGENT_MODEL_ROOT="$ASSET_ROOT/models/Qwen3-8B"
EMBED_MODEL_ROOT="$ASSET_ROOT/models/Qwen3-Embedding-0.6B"

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== RARG claim-by-claim reproduction ==="
echo "backend=kubernetes"
echo "gpu_model=NVIDIA RTX PRO 6000 Blackwell"
echo "allocated_gpus=4"
echo "commit=$(git rev-parse HEAD)"
echo "run_command=bash reproduction/run_k8s.sh"
echo "config=$(python -c 'import json; print(json.dumps(json.load(open("reproduction/config.json")), sort_keys=True))')"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -m pip install --quiet --disable-pip-version-check \
  "transformers>=4.54,<5" "sentence-transformers>=5,<6" \
  "huggingface-hub>=0.34,<2" "faiss-cpu>=1.11,<2" \
  "pyarrow>=18,<22" "numpy<3" "scipy<2" "openai>=1,<3"
python -m pip install --quiet --disable-pip-version-check -e "$REPO_ROOT"

apt-get update -qq
apt-get install -y -qq ripgrep >/dev/null
command -v rg
rg --version | head -1

mkdir -p "$ASSET_ROOT" "$HF_CACHE"

(
  flock -x 9
  if [ ! -f "$ASSET_ROOT/.assets-complete" ]; then
    echo "ASSET_PREP_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    mkdir -p "$(dirname "$PARQUET_PATH")" "$ASSET_ROOT/models"

    if [ ! -f "$PARQUET_PATH" ]; then
      hf download DCI-Agent/corpus browsecomp_plus/data.parquet \
        --repo-type dataset --local-dir "$ASSET_ROOT"
    fi

    if [ ! -d "$CORPUS_ROOT" ] || [ "$(find "$CORPUS_ROOT" -type f | wc -l)" -lt 100000 ]; then
      rm -rf "$CORPUS_ROOT"
      python -m dci.benchmark.export_bc_plus_docs \
        --source-dir "$(dirname "$PARQUET_PATH")" \
        --output-dir "$CORPUS_ROOT"
    fi

    if [ ! -f "$AGENT_MODEL_ROOT/config.json" ]; then
      hf download Qwen/Qwen3-8B --local-dir "$AGENT_MODEL_ROOT"
    fi
    if [ ! -f "$EMBED_MODEL_ROOT/config.json" ]; then
      hf download Qwen/Qwen3-Embedding-0.6B --local-dir "$EMBED_MODEL_ROOT"
    fi

    if [ ! -f "$INDEX_ROOT/index.faiss" ]; then
      rm -rf "$INDEX_ROOT"
      python "$REPO_ROOT/scripts/build_embedding_index.py" \
        --corpus-dir "$CORPUS_ROOT" \
        --output-dir "$INDEX_ROOT" \
        --model "$EMBED_MODEL_ROOT" \
        --model-type qwen3_embedding_4b \
        --backend transformers \
        --device cuda:0 \
        --max-chars 6000 \
        --max-model-len 512 \
        --batch-size 96
    fi

    python "$REPO_ROOT/reproduction/build_docid_map.py" \
      --parquet "$PARQUET_PATH" \
      --paths "$INDEX_ROOT/paths.json" \
      --output "$INDEX_ROOT/docid_to_path.json"

    touch "$ASSET_ROOT/.assets-complete"
    echo "ASSET_PREP_END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "ASSET_CACHE_HIT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
) 9>"$SHARED_ROOT/assets.lock"

python "$REPO_ROOT/reproduction/run_reproduction.py" \
  --config "$REPO_ROOT/reproduction/config.json" \
  --dataset "$REPO_ROOT/data/bcplus_qa_sample100.jsonl" \
  --corpus "$CORPUS_ROOT" \
  --index "$INDEX_ROOT" \
  --docid-map "$INDEX_ROOT/docid_to_path.json" \
  --agent-model "$AGENT_MODEL_ROOT" \
  --embedding-model "$EMBED_MODEL_ROOT"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
