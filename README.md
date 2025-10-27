# Nanochat Ruby

Ruby port of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy. Loads PyTorch checkpoints for inference and fine-tuning.

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

The `bin/speedrun.sh` script trains a tiny checkpoint using [python-nanochat](https://github.com/karpathy/nanochat). Ruby nanochat loads it for inference.

## Fine-Tuning

```bash
ruby examples/finetune.rb \
  --data my_training_data.txt \
  --epochs 3 \
  --output custom_model.pt
```

Uses AdamW with gradient clipping. Saves best checkpoint. See `examples/sample_training_data.txt` for data format.

## What Works

**Complete:**
- Inference: CLI chat, web UI, streaming, KV caching
- Fine-tuning on single GPU
- Tokenizer training
- PyTorch checkpoint loading

**Not implemented:**
- Multi-GPU training (not planned)
- Evaluation benchmarks (maybe later)
- Base training from scratch (maybe later)
- Reinforcement learning (maybe later)

## How It Works

GPT model with RoPE, GQA and causal attention. BPE tokenizer via HuggingFace. KV cache for fast inference. Temperature, top-k and top-p sampling. Auto-detects CUDA, MPS or CPU.

## Requirements

Ruby >= 3.4.0. LibTorch installs automatically via torch-rb.

## Tokenizer Training

```ruby
require 'nanochat'

tokenizer = Nanochat::Tokenizer.train_from_files(
  ['data/train.txt'],
  vocab_size: 50_257
)
tokenizer.save('~/.cache/nanochat/my_tokenizer')
```

Tokenizers trained in Python work in Ruby (both save as `.json`). RustBPE `.pkl` files don't work.

## Benchmarks

```bash
ruby examples/benchmark.rb
```

Benchmarks tokenizer, forward pass, generation and sampling.

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

## Training Models

`bin/speedrun.sh` trains a tiny d4 model (~30 mins on CPU).

For production d20 models, use [python-nanochat](https://github.com/karpathy/nanochat) (~4 hours on 8×H100):

```bash
git clone https://github.com/karpathy/nanochat
cd nanochat
bash speedrun.sh
```

Ruby nanochat loads any checkpoint from python-nanochat.

## Contributing

Pull requests welcome. Priorities: evaluation benchmarks, base training, performance.

## Credits

Andrej Karpathy for nanochat. Andrew Kane for torch-rb and tokenizers-ruby.

## Links

- [python-nanochat](https://github.com/karpathy/nanochat)
- [torch-rb](https://github.com/ankane/torch-rb)
- [tokenizers-ruby](https://github.com/ankane/tokenizers-ruby)
