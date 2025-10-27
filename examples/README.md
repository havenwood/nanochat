# Examples

Train a model first:
```bash
bash bin/speedrun.sh
```

Then run examples:

```bash
# Chat
ruby examples/chat_cli.rb

# Web UI
ruby examples/chat_web.rb

# Text generation
ruby examples/generate_text.rb "Once upon a time"

# Fine-tuning
ruby examples/finetune.rb --data sample_training_data.txt

# Benchmarks
ruby examples/benchmark.rb
```

All examples use `~/.cache/nanochat/model.pt` by default.
