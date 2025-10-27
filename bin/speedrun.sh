#!/bin/bash

# Train a tiny nanochat checkpoint for Ruby nanochat
#
# This is a simplified version adapted for CPU/single-GPU training.
# For the full $100 8×H100 training, use python-nanochat directly.
#
# This script trains a d4 model (4 layers, ~30 mins on CPU)
# The Python nanochat speedrun.sh trains a d20 model (~4 hours on 8×H100)

set -e

echo "======================================================================"
echo "🚀 Ruby Nanochat - Quick Training Demo"
echo "======================================================================"
echo ""
echo "This trains a tiny d4 model suitable for testing Ruby nanochat."
echo ""
echo "📊 Model specs:"
echo "  • 4 layers (vs 20 in Python speedrun.sh)"
echo "  • Runs on CPU or single GPU"
echo "  • ~30 minutes training time"
echo "  • Good for demos and development"
echo ""
echo "💡 For production models, use python-nanochat/speedrun.sh directly"
echo "   (requires 8×H100 GPUs, ~4 hours, ~$100)"
echo ""
echo "======================================================================"
echo ""

# Run the actual training script
bash bin/train-with-python-nanochat.sh
