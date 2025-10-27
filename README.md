# 💬 Nanochat Ruby

> The simplest way to train and deploy language models in Ruby.

Ruby implementation of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy. Inference, fine-tuning, and chat applications with full PyTorch checkpoint compatibility.

[![Ruby](https://img.shields.io/badge/ruby-%3E%3D%203.4.0-red.svg)](https://www.ruby-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Quick Start

Train and chat with your own language model:

```bash
# Train a demo model (~30 mins on CPU)
bash bin/speedrun.sh

# Install Ruby gem (prerelease)
gem install nanochat --pre

# Chat with your model
ruby examples/chat_cli.rb
```

That's it! The `bin/speedrun.sh` script trains a tiny checkpoint using [python-nanochat](https://github.com/karpathy/nanochat)'s training code. Ruby nanochat then loads the checkpoint for inference.

## ⚠️ Prerelease Status

This is a **prerelease** version (`0.1.0.pre`) and does not yet have full parity with the [Python nanochat](https://github.com/karpathy/nanochat) implementation. Core inference and fine-tuning features work, but some advanced features are still in development. See the Feature Comparison table below for details.

## Fine-Tuning

Adapt pre-trained models to your domain with a single GPU:

```bash
ruby examples/finetune.rb \
  --data my_training_data.txt \
  --epochs 3 \
  --output custom_model.pt
```

Features:
- AdamW optimizer with gradient clipping
- Automatic checkpointing (saves best model)
- Plain text data format (see `examples/sample_training_data.txt`)

## Feature Comparison

| Feature | Python | Ruby | Status |
|---------|--------|-------------|--------|
| **Inference** | | | |
| CLI chat | ✅ | ✅ | Complete |
| Web UI | ✅ | ✅ | Complete |
| Streaming generation | ✅ | ✅ | Complete |
| KV caching | ✅ | ✅ | Complete |
| **Training** | | | |
| Single-GPU base training | ✅ | ⚠️ | Maybe later |
| Multi-GPU (8×H100) | ✅ | ❌ | Not planned |
| Fine-tuning | ✅ | ✅ | Complete |
| Reinforcement learning | ✅ | ❌ | Maybe later |
| **Evaluation** | | | |
| MMLU, ARC, GSM8K | ✅ | ⚠️ | Maybe later |
| HumanEval | ✅ | ⚠️ | Maybe later |
| **Data** | | | |
| Tokenizer training | ✅ | ✅ | Complete |
| Checkpoint loading | ✅ | ✅ | Complete |

**Legend**: ✅ Complete | ⚠️ Maybe later | ❌ Not planned

## Architecture

Nanochat Ruby implements:
- **GPT Model**: Decoder-only transformer with RoPE, GQA, causal attention
- **Tokenizer**: HuggingFace tokenizers gem (BPE encoding)
- **Engine**: Efficient inference with KV cache
- **Sampling**: Temperature, top-k, top-p (nucleus)
- **Device Support**: Auto-detection of CUDA, MPS (Apple Silicon), CPU

## Requirements

- **Ruby** >= 3.4.0
- **LibTorch** (installed automatically via torch-rb)
- **HuggingFace tokenizers** gem

## Training Tokenizers

Train BPE tokenizers in Ruby:

```ruby
require 'nanochat'

tokenizer = Nanochat::Tokenizer.train_from_files(
  ['data/train.txt'],
  vocab_size: 50_257
)
tokenizer.save('~/.cache/nanochat/my_tokenizer')
```

Or train in Python and use in Ruby (both create `.json` format):

```python
from nanochat.tokenizer import HuggingFaceTokenizer
tokenizer = HuggingFaceTokenizer.train_from_iterator(text, vocab_size=50_257)
tokenizer.save("~/.cache/nanochat/tokenizer")
```

⚠️ **Note**: RustBPETokenizer `.pkl` files are not compatible with Ruby.

## Performance Benchmarks

Measure inference performance:

```bash
ruby examples/benchmark.rb
```

Benchmarks tokenizer, model forward pass, generation throughput, and sampling strategies. Uses `benchmark-ips` for statistical analysis.

## Development

```bash
# Clone repository
git clone https://github.com/havenwood/nanochat
cd nanochat

# Install dependencies
bundle install

# Run tests
bundle exec rake test

# Run linter
bundle exec rubocop

# Run all checks
bundle exec rake

# Run performance benchmarks
ruby examples/benchmark.rb
```

## Current Features

**Available Now** ✅:
- Inference with all sampling strategies (temperature, top-k, top-p)
- Fine-tuning on pre-trained models (single-GPU)
- Tokenizer training from text files
- CLI and web UI examples with streaming
- Performance benchmarks

**Maybe later** ⚠️:
- Evaluation benchmarks (MMLU, ARC, GSM8K, HumanEval)
- Single-GPU base training from scratch

## Training Your Own Models

Ruby nanochat's `bin/speedrun.sh` trains a **tiny demo model** (d4, ~30 mins on CPU) for quick testing.

For **production models**, use [python-nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy:

```bash
git clone https://github.com/karpathy/nanochat
cd nanochat
bash speedrun.sh  # Production d20 model: ~4 hours on 8×H100, ~$100
```

Then use your checkpoint in Ruby:
```bash
ruby examples/chat_cli.rb
ruby examples/chat_web.rb
```

Ruby nanochat loads any PyTorch checkpoint trained with python-nanochat.

## Contributing

Contributions welcome! Priority areas:
- Evaluation benchmark implementations (MMLU, ARC, GSM8K, HumanEval)
- Single-GPU base training from scratch
- Performance optimizations
- Integration tests with real trained models

## Acknowledgments

- **[Andrej Karpathy](https://github.com/karpathy)** for the original nanochat
- **[Andrew Kane](https://github.com/ankane)** for torch-rb and tokenizers-ruby

## Links

- **Python nanochat**: https://github.com/karpathy/nanochat
- **torch-rb**: https://github.com/ankane/torch-rb
- **HuggingFace Tokenizers**: https://github.com/ankane/tokenizers-ruby
