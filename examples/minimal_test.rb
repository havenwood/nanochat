#!/usr/bin/env ruby
# frozen_string_literal: true

# Minimal test of Ruby nanochat architecture without requiring a trained model
#
# This creates a tiny random model to verify that all components work together.
# This is NOT for real text generation - just architecture validation.

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'

# Mock tokenizer for minimal testing (doesn't need real BPE)
class MockTokenizer
  attr_reader :vocab_size, :bos_token_id, :eos_token_id

  def initialize(vocab_size)
    @vocab_size = vocab_size
    @bos_token_id = 0
    @eos_token_id = 1
  end

  def encode(text, prepend: nil, append: nil)
    # Convert text to token IDs (simple: use character codes mod vocab_size)
    ids = text.bytes.map { |byte| byte % @vocab_size }
    ids.unshift(prepend) if prepend
    ids << append if append
    ids
  end

  def decode(ids)
    # Simple decode: map IDs back to characters
    ids.map { |id| (id % 128).chr }.join
  rescue StandardError
    ids.map(&:to_s).join(' ')
  end

  def token_to_id(_token)
    nil # Mock doesn't have special tokens
  end

  def id_to_token(token_id)
    token_id.to_s
  end
end

# rubocop:disable Naming/PredicateMethod
def validate_nanochat_architecture
  # rubocop:enable Naming/PredicateMethod
  puts '🧪 Ruby Nanochat Architecture Test'
  puts '=' * 50
  puts

  # Create a minimal configuration
  config = Nanochat::Config.new(
    vocab_size: 512,   # Tiny vocab
    n_embd: 64,        # Tiny model
    n_layer: 2,        # Just 2 layers
    n_head: 4,         # 4 attention heads
    n_kv_head: 2,      # 2 KV heads (GQA)
    block_size: 64     # Context length
  )

  puts '📋 Test Configuration:'
  puts "  Vocab size: #{config.vocab_size}"
  puts "  Embedding dim: #{config.n_embd}"
  puts "  Layers: #{config.n_layer}"
  puts "  Query heads: #{config.n_head}"
  puts "  KV heads: #{config.n_kv_head}"
  puts "  Block size: #{config.block_size}"
  puts

  # Create model with random weights
  puts '🔄 Creating model with random weights...'
  model = Nanochat::GPT.new(config)
  puts '✅ Model created successfully'
  puts

  # Create a mock tokenizer for testing
  # (A real tokenizer would need to be properly trained)
  puts '🔄 Creating mock tokenizer...'
  tokenizer = MockTokenizer.new(config.vocab_size)
  puts "✅ Mock tokenizer created (vocab size: #{tokenizer.vocab_size})"
  puts

  # Create engine
  puts '🔄 Creating inference engine...'
  device = Nanochat::Common.device
  puts "  Device: #{device}"
  engine = Nanochat::Engine.new(model:, tokenizer:, device:)
  puts '✅ Engine created'
  puts

  # Test forward pass
  puts '🧪 Testing forward pass...'
  begin
    # Encode a simple prompt
    prompt = 'Test'
    tokens = tokenizer.encode(prompt)
    puts "  Input: \"#{prompt}\""
    puts "  Tokens: #{tokens.inspect}"

    # Run forward pass
    input_tensor = Torch.tensor([tokens], dtype: :long)
    input_tensor = input_tensor.to(device)

    logits = model.call(input_tensor)
    puts "  Output shape: #{logits.shape.inspect}"
    puts '  ✅ Forward pass successful'
    puts
  rescue StandardError => e
    puts "  ❌ Forward pass failed: #{e.message}"
    puts e.backtrace.first(5)
    return false
  end

  # Test generation (with random weights, output will be nonsense)
  puts '🧪 Testing generation...'
  begin
    response = engine.generate(
      'Test',
      max_tokens: 10,
      temperature: 1.0
    )
    puts "  Generated: \"#{response}\""
    puts '  ✅ Generation successful (output is random due to untrained weights)'
    puts
  rescue StandardError => e
    puts "  ❌ Generation failed: #{e.message}"
    puts e.backtrace.first(5)
    return false
  end

  # Test sampling strategies
  puts '🧪 Testing sampling strategies...'
  strategies = [
    {name: 'Temperature 0.5', temperature: 0.5},
    {name: 'Top-k 10', temperature: 1.0, top_k: 10},
    {name: 'Top-p 0.9', temperature: 1.0, top_p: 0.9}
  ]

  strategies.each do |strategy|
    engine.generate('Test', max_tokens: 5, **strategy.except(:name))
    puts "  ✅ #{strategy[:name]}"
  rescue StandardError => e
    puts "  ❌ #{strategy[:name]}: #{e.message}"
    return false
  end
  puts

  # All tests passed
  puts '=' * 50
  puts '✅ All architecture tests passed!'
  puts
  puts '📊 Summary:'
  puts '  ✅ Model initialization'
  puts '  ✅ Tokenizer integration'
  puts '  ✅ Forward pass'
  puts '  ✅ Generation with KV cache'
  puts '  ✅ Sampling strategies'
  puts
  puts '🎯 Next: Test with a real trained model'
  puts '   Run: ruby examples/integration_test.rb'
  puts

  true
end

(validate_nanochat_architecture && exit(0)) || exit(1) if $PROGRAM_NAME == __FILE__
