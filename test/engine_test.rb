# frozen_string_literal: true

require 'test_helper'

class EngineTest < Minitest::Test
  def setup
    @config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 4,
      n_kv_head: 4,
      n_layer: 2
    )
    @model = Nanochat::GPT.new(@config)
    @tokenizer = create_mock_tokenizer
  end

  def test_engine_initialization
    engine = Nanochat::Engine.new(model: @model, tokenizer: @tokenizer)

    assert_kind_of Nanochat::Engine, engine
  end

  def test_engine_initialization_with_custom_device
    device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: device)

    assert_kind_of Nanochat::Engine, engine
  end

  def test_generate_returns_string
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('Hello', max_tokens: 5)

    assert_kind_of String, output
  end

  def test_generate_with_max_tokens
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('Test', max_tokens: 3)

    assert_kind_of String, output
  end

  def test_generate_with_temperature
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('Test', max_tokens: 3, temperature: 0.8)

    assert_kind_of String, output
  end

  def test_generate_with_top_k
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('Test', max_tokens: 3, top_k: 50)

    assert_kind_of String, output
  end

  def test_generate_with_top_p
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('Test', max_tokens: 3, top_p: 0.9)

    assert_kind_of String, output
  end

  def test_top_p_nucleus_sampling
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    logits = Torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    samples = []
    10.times do
      sample = engine.send(:sample, logits, 1.0, nil, 0.9)
      samples << sample[0, 0].item
    end

    refute_includes samples, 0, 'Top-p sampling should filter out lowest probability token'
  end

  def test_generate_with_empty_prompt
    cpu_device = Torch.device('cpu')
    model = Nanochat::GPT.new(@config)
    engine = Nanochat::Engine.new(model: model, tokenizer: @tokenizer, device: cpu_device)

    output = engine.generate('', max_tokens: 3)

    assert_kind_of String, output
  end

  private

  # Create a simple mock tokenizer for testing
  def create_mock_tokenizer
    # Create a simple word-level tokenizer
    underlying = Tokenizers::Tokenizer.new(
      Tokenizers::Models::WordLevel.new(vocab: {}, unk_token: '[UNK]')
    )

    # Simple whitespace splitter
    underlying.pre_tokenizer = Tokenizers::PreTokenizers::Whitespace.new

    # Train on a small corpus to build vocabulary
    trainer = Tokenizers::Trainers::WordLevelTrainer.new(
      vocab_size: 1000,
      min_frequency: 1
    )

    # Create a temporary file with sample text
    require 'tempfile'
    Tempfile.create(['sample', '.txt']) do |file|
      file.write('Hello world Test sample text for tokenizer training ' * 100)
      file.flush
      underlying.train([file.path], trainer)
    end

    Nanochat::Tokenizer.new(underlying)
  end
end
