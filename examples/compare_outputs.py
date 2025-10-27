#!/usr/bin/env python3
"""
Compare Python nanochat output with Ruby nanochat output.

This script runs the same prompts through both Python and Ruby implementations
to verify they produce similar results.

Usage:
    python examples/compare_outputs.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Add python-nanochat to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python-nanochat"))

from nanochat.engine import Engine
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_device

def generate_python(prompt, max_tokens=50, temperature=0.8):
    """Generate text using Python nanochat."""
    # Load model
    checkpoint_path = Path.home() / ".cache" / "nanochat" / "model.pt"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load model
    model = GPT.from_checkpoint(str(checkpoint_path))

    # Create engine
    device = get_device()
    engine = Engine(model, tokenizer, device=device)

    # Generate
    return engine.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95
    )

def generate_ruby(prompt, max_tokens=50, temperature=0.8):
    """Generate text using Ruby nanochat."""
    script = f"""
require 'bundler/setup'
require 'nanochat'

checkpoint_path = File.expand_path('~/.cache/nanochat/model.pt')
tokenizer_dir = File.expand_path('~/.cache/nanochat/tokenizer')

config = Nanochat::Config.new(
  vocab_size: 50_257,
  n_embd: 768,
  n_layer: 20,
  n_head: 12,
  n_kv_head: 4,
  block_size: 1024
)

model = Nanochat::GPT.from_checkpoint(checkpoint_path, config)
tokenizer = Nanochat::Tokenizer.from_directory(tokenizer_dir)
engine = Nanochat::Engine.new(model, tokenizer)

response = engine.generate(
  {repr(prompt)},
  max_tokens: {max_tokens},
  temperature: {temperature},
  top_p: 0.95
)

puts response
"""

    try:
        result = subprocess.run(
            ["ruby", "-e", script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"❌ Ruby error: {result.stderr}")
            return None

        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("❌ Ruby generation timed out")
        return None
    except Exception as e:
        print(f"❌ Ruby execution failed: {e}")
        return None

def compare_outputs():
    """Compare Python and Ruby outputs for multiple prompts."""
    test_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "Hello, how are you?",
    ]

    print("🔬 Python vs Ruby Nanochat Comparison")
    print("=" * 70)
    print()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: \"{prompt}\"")
        print("-" * 70)

        # Generate with Python
        print("🐍 Python output:")
        python_output = generate_python(prompt, max_tokens=50, temperature=0.8)
        if python_output:
            print(f"  {python_output}")
        else:
            print("  (failed)")
        print()

        # Generate with Ruby
        print("💎 Ruby output:")
        ruby_output = generate_ruby(prompt, max_tokens=50, temperature=0.8)
        if ruby_output:
            print(f"  {ruby_output}")
        else:
            print("  (failed)")
        print()

        # Analysis
        if python_output and ruby_output:
            # Check if outputs start the same
            common_prefix = os.path.commonprefix([python_output, ruby_output])
            if len(common_prefix) > len(prompt):
                print(f"✅ Outputs share common prefix ({len(common_prefix)} chars)")
            else:
                print(f"⚠️  Outputs diverge immediately")

            # Token-level comparison would be better but requires tokenizer
            print(f"  Python length: {len(python_output)} chars")
            print(f"  Ruby length: {len(ruby_output)} chars")

        print()
        print("=" * 70)
        print()

if __name__ == "__main__":
    try:
        compare_outputs()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
