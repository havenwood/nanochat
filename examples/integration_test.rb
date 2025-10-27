#!/usr/bin/env ruby
# frozen_string_literal: true

# Integration test for Ruby nanochat with real trained models
#
# This script demonstrates end-to-end inference with a trained checkpoint.
# To use this script:
#
# Option 1: Download pre-trained checkpoint (easiest)
#    nanochat-setup
#
# Option 2: Train your own model using python-nanochat
#    git clone https://github.com/karpathy/nanochat
#    cd nanochat && bash speedrun.sh
#
# Then run this script:
#    ruby examples/integration_test.rb

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'

# Configuration
CHECKPOINT_PATH = File.expand_path('~/.cache/nanochat/model.pt')
TOKENIZER_DIR = File.expand_path('~/.cache/nanochat/tokenizer')

def main
  puts '🚀 Ruby Nanochat Integration Test'
  puts '=' * 50

  # Check files exist
  unless File.exist?(CHECKPOINT_PATH)
    puts "❌ Checkpoint not found: #{CHECKPOINT_PATH}"
    puts "\nOption 1: Download pre-trained checkpoint"
    puts '  nanochat-setup'
    puts "\nOption 2: Train your own with python-nanochat"
    puts '  git clone https://github.com/karpathy/nanochat'
    puts '  cd nanochat && bash bin/speedrun.sh'
    exit 1
  end

  tokenizer_file = File.join(TOKENIZER_DIR, 'tokenizer.json')
  unless File.exist?(tokenizer_file)
    puts "❌ Tokenizer not found: #{tokenizer_file}"
    puts "\nRun nanochat-setup to download both model and tokenizer:"
    puts '  nanochat-setup'
    exit 1
  end

  puts "✅ Found checkpoint: #{CHECKPOINT_PATH}"
  puts "✅ Found tokenizer: #{tokenizer_file}"
  puts

  # Load configuration
  # These values should match your trained model
  # For speedrun.sh (d20): n_embd=768, n_layer=20
  # For run1000.sh (d32): n_embd=1024, n_layer=32
  config = Nanochat::Config.new(
    vocab_size: 50_257,  # Adjust if you used different vocab size
    n_embd: 768,         # Adjust based on your model
    n_layer: 20,         # Adjust based on your model
    n_head: 12,          # Adjust based on your model
    n_kv_head: 4,        # Adjust based on your model (GQA)
    block_size: 1024     # Context length
  )

  puts '📋 Model Configuration:'
  puts "  Vocab size: #{config.vocab_size}"
  puts "  Embedding dim: #{config.n_embd}"
  puts "  Layers: #{config.n_layer}"
  puts "  Query heads: #{config.n_head}"
  puts "  KV heads: #{config.n_kv_head}"
  puts "  Block size: #{config.block_size}"
  puts

  # Load model
  puts '🔄 Loading model...'
  model = Nanochat::GPT.from_checkpoint(CHECKPOINT_PATH, config)
  puts '✅ Model loaded successfully'
  puts

  # Load tokenizer
  puts '🔄 Loading tokenizer...'
  tokenizer = Nanochat::Tokenizer.from_directory(TOKENIZER_DIR)
  puts "✅ Tokenizer loaded (vocab size: #{tokenizer.vocab_size})"
  puts

  # Create engine
  puts '🔄 Creating inference engine...'
  device = Nanochat::Common.device
  puts "  Device: #{device}"
  engine = Nanochat::Engine.new(model:, tokenizer:, device:)
  puts '✅ Engine ready'
  puts

  # Run test prompts
  test_prompts = [
    'Once upon a time',
    'The meaning of life is',
    'In a world where',
    'Hello, how are you?'
  ]

  puts '🎯 Running test generations...'
  puts '=' * 50
  puts

  test_prompts.each_with_index do |prompt, index|
    puts "Test #{index + 1}/#{test_prompts.length}"
    puts "Prompt: \"#{prompt}\""
    puts 'Generating (max_tokens: 50, temperature: 0.8)...'
    puts

    begin
      response = engine.generate(
        prompt,
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95
      )

      puts "Response: #{response}"
      puts
      puts '-' * 50
      puts
    rescue StandardError => e
      puts "❌ Generation failed: #{e.message}"
      puts e.backtrace.first(5)
      puts
    end
  end

  puts '✅ Integration test complete!'
  puts
  puts '📊 Next Steps:'
  puts '  1. Compare outputs with Python nanochat'
  puts '  2. Test with different sampling parameters'
  puts '  3. Verify numerical accuracy'
  puts '  4. Test with longer sequences'
end

main if $PROGRAM_NAME == __FILE__
