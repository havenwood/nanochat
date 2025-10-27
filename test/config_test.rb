# frozen_string_literal: true

require 'test_helper'

class ConfigTest < Minitest::Test
  def setup
    @valid_config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 4,
      n_kv_head: 4,
      n_layer: 2
    )
  end

  def test_config_creation
    assert_equal 1000, @valid_config.vocab_size
    assert_equal 128, @valid_config.block_size
    assert_equal 64, @valid_config.n_embd
    assert_equal 4, @valid_config.n_head
    assert_equal 4, @valid_config.n_kv_head
    assert_equal 2, @valid_config.n_layer
  end

  def test_config_is_data = assert_kind_of(Data, @valid_config)

  def test_config_default_values
    config = Nanochat::Config.default

    assert_equal 50_304, config.vocab_size
    assert_equal 1024, config.block_size
    assert_equal 768, config.n_embd
    assert_equal 6, config.n_head
    assert_equal 6, config.n_kv_head
    assert_equal 12, config.n_layer
  end

  def test_config_default_validates = assert_nil(Nanochat::Config.default.validate!)

  def test_config_validation_passes
    assert_nil @valid_config.validate!
  end

  def test_config_validation_fails_for_invalid_heads
    config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 5, # 64 not divisible by 5
      n_kv_head: 5,
      n_layer: 2
    )

    error = assert_raises(ArgumentError) { config.validate! }
    assert_match(/n_embd.*divisible.*n_head/i, error.message)
  end

  def test_config_validation_fails_for_zero_heads
    config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 0,
      n_kv_head: 0,
      n_layer: 2
    )

    assert_raises(ArgumentError) { config.validate! }
  end

  def test_config_validation_fails_for_invalid_mqa
    config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 4,
      n_kv_head: 5,  # Must be <= n_head
      n_layer: 2
    )

    error = assert_raises(ArgumentError) { config.validate! }
    assert_match(/n_kv_head.*n_head/i, error.message)
  end

  def test_config_validation_fails_when_n_head_not_divisible_by_n_kv_head
    config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 8,
      n_kv_head: 3,  # 8 not divisible by 3
      n_layer: 2
    )

    error = assert_raises(ArgumentError) { config.validate! }
    assert_match(/n_head.*divisible.*n_kv_head/i, error.message)
  end

  def test_config_with_minimal_values
    config = Nanochat::Config.new(
      vocab_size: 256,
      block_size: 64,
      n_embd: 32,
      n_head: 2,
      n_kv_head: 1,
      n_layer: 1
    )

    assert_equal 256, config.vocab_size
    assert_equal 1, config.n_layer
    assert_nil config.validate!
  end

  def test_config_with_large_model_values
    config = Nanochat::Config.new(
      vocab_size: 100_000,
      block_size: 2048,
      n_embd: 1024,
      n_head: 16,
      n_kv_head: 16,
      n_layer: 24
    )

    assert_equal 100_000, config.vocab_size
    assert_equal 2048, config.block_size
    assert_nil config.validate!
  end

  def test_config_with_grouped_query_attention
    config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 8,
      n_kv_head: 2,  # GQA: 4 query heads per KV head
      n_layer: 2
    )

    assert_equal 8, config.n_head
    assert_equal 2, config.n_kv_head
    assert_nil config.validate!
  end

  def test_head_dimension_calculation
    # n_embd / n_head should give head dimension
    assert_equal 16, @valid_config.n_embd / @valid_config.n_head
  end
end
