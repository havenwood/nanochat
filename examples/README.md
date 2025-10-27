# Nanochat Examples

This directory contains usage examples for nanochat.

## Prerequisites

1. Install the gem:
   ```bash
   gem install nanochat
   ```

2. Download a model checkpoint from [nanochat](https://github.com/karpathy/nanochat)

## Examples

### Chat CLI

Interactive chat interface:

```bash
ruby examples/chat_cli.rb path/to/checkpoint.pt
```

### Generate Text

Simple text generation:

```bash
ruby examples/generate_text.rb path/to/checkpoint.pt "Your prompt here"
```

## Notes

- All examples expect a PyTorch checkpoint file (`.pt`) trained with python-nanochat
- GPU support is automatic if CUDA or MPS is available
- Generation parameters can be adjusted in the code
