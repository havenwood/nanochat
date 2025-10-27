#!/bin/bash

# Train a tiny nanochat checkpoint using Python nanochat
#
# This script is adapted from python-nanochat/dev/runcpu.sh
# Original: https://github.com/karpathy/nanochat by Andrej Karpathy
#
# REQUIREMENTS:
#   - Python nanochat cloned at: ../python-nanochat or ./python-nanochat
#   - Python 3.10+
#   - Rust (for building rustbpe tokenizer)
#   - ~1GB disk space for data
#   - ~30 minutes on CPU or ~5 minutes on GPU
#
# USAGE:
#   bash bin/train-with-python-nanochat.sh
#
# OUTPUT:
#   Trained checkpoint at: ~/.cache/nanochat/model.pt
#   Tokenizer at: ~/.cache/nanochat/tokenizer/tokenizer.json

set -e  # Exit on error

echo "🔥 Train Tiny Nanochat Model"
echo "======================================================================"
echo ""
echo "This will train a d4 model (4 layers, minimal for demos)"
echo "Output: ~/.cache/nanochat/"
echo ""
echo "⏱️  Estimated time: ~30 minutes on CPU"
echo ""
echo "📝 Attribution: Using training scripts from"
echo "   https://github.com/karpathy/nanochat by Andrej Karpathy"
echo ""
echo "======================================================================"
echo ""

# Find python-nanochat directory
if [ -d "python-nanochat" ]; then
    PYTHON_NANOCHAT_DIR="python-nanochat"
elif [ -d "../python-nanochat" ]; then
    PYTHON_NANOCHAT_DIR="../python-nanochat"
else
    echo "❌ Python nanochat not found"
    echo ""
    echo "Clone it first:"
    echo "  git clone https://github.com/karpathy/nanochat python-nanochat"
    echo ""
    exit 1
fi

echo "✅ Python nanochat found at: $PYTHON_NANOCHAT_DIR"
echo ""

# Change to python-nanochat directory
cd "$PYTHON_NANOCHAT_DIR"

# Setup environment
export OMP_NUM_THREADS=1
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

echo "🔧 Setting up Python environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
echo ""

echo "🦀 Building Rust tokenizer..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
echo ""

echo "📦 Downloading evaluation bundle..."
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip "$EVAL_BUNDLE_URL"
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle "$NANOCHAT_BASE_DIR"
fi
echo ""

# Reset report
python -m nanochat.report reset

echo "📚 Downloading training data (~1GB)..."
python -m nanochat.dataset -n 4
echo ""

echo "🔤 Training tokenizer..."
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval
echo ""

echo "🚀 Training base model (50 iterations, ~30 mins)..."
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50
echo ""

echo "📊 Evaluating base model..."
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16
echo ""

echo "🎯 Midtraining (100 iterations)..."
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100
echo ""

echo "💬 Supervised fine-tuning (100 iterations)..."
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16
echo ""

echo "📝 Generating training report..."
python -m nanochat.report generate
echo ""

echo "======================================================================"
echo "✅ Training complete!"
echo ""
echo "📦 Checkpoint location:"
echo "  Model: $NANOCHAT_BASE_DIR/model.pt"
echo "  Tokenizer: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
echo ""
echo "🎯 Next Steps - Use Your Model in Ruby"
echo "======================================================================"
echo ""
echo "# Interactive chat"
echo "ruby examples/chat_cli.rb"
echo ""
echo "# Web UI (visit http://localhost:8000)"
echo "ruby examples/chat_web.rb"
echo ""
echo "# Generate text"
echo "ruby examples/generate_text.rb 'Once upon a time'"
echo ""
echo "# Fine-tune on your data"
echo "ruby examples/finetune.rb --data my_data.txt --output custom.pt"
echo ""
echo "======================================================================"
echo ""
echo "📦 Optional: Package this checkpoint for distribution"
echo ""
echo "tar -czf nanochat-tiny-d4.tar.gz -C $(dirname $NANOCHAT_BASE_DIR) $(basename $NANOCHAT_BASE_DIR)"
echo ""
