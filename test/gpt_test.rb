# frozen_string_literal: true

require 'test_helper'

class GPTTest < Minitest::Test
  def setup
    @config = Nanochat::Config.new(
      vocab_size: 1000,
      block_size: 128,
      n_embd: 64,
      n_head: 4,
      n_kv_head: 4,
      n_layer: 2
    )
  end

  def test_gpt_initialization
    model = Nanochat::GPT.new(@config)

    assert_kind_of Nanochat::GPT, model
  end

  def test_gpt_is_torch_module = assert_kind_of(Torch::NN::Module, Nanochat::GPT.new(@config))

  def test_gpt_forward_shape
    model = Nanochat::GPT.new(@config)
    input_ids = Torch.randint(0, 1000, [2, 10], dtype: :long) # batch=2, seq_len=10

    output = model.forward(input_ids)

    # Should return [batch, seq_len, vocab_size]
    assert_equal [2, 10, 1000], output.size.to_a
  end

  def test_gpt_forward_single_batch
    model = Nanochat::GPT.new(@config)
    input_ids = Torch.randint(0, 1000, [1, 5], dtype: :long)

    output = model.forward(input_ids)

    assert_equal [1, 5, 1000], output.size.to_a
  end

  def test_gpt_forward_with_targets_returns_loss
    model = Nanochat::GPT.new(@config)
    input_ids = Torch.randint(0, 1000, [2, 10], dtype: :long)
    targets = Torch.randint(0, 1000, [2, 10], dtype: :long)

    loss = model.forward(input_ids, targets: targets)

    assert_kind_of Torch::Tensor, loss
    assert_equal 0, loss.ndim # Scalar loss
  end

  def test_gpt_allows_sequences_within_rotary_limit
    model = Nanochat::GPT.new(@config)
    # RoPE embeddings are pre-allocated to 10x block_size
    long_seq = Torch.randint(0, 1000, [1, @config.block_size + 1], dtype: :long)

    # Should work fine since rotary_seq_len = block_size * 10
    output = model.forward(long_seq)

    assert_equal [1, @config.block_size + 1, 1000], output.size.to_a
  end

  def test_gpt_with_minimal_config
    small_config = Nanochat::Config.new(
      vocab_size: 256,
      block_size: 64,
      n_embd: 32,
      n_head: 2,
      n_kv_head: 1,
      n_layer: 1
    )
    model = Nanochat::GPT.new(small_config)
    input_ids = Torch.randint(0, 256, [1, 10], dtype: :long)

    output = model.forward(input_ids)

    assert_equal [1, 10, 256], output.size.to_a
  end

  def test_gpt_rotary_embeddings_precomputed
    model = Nanochat::GPT.new(@config)
    cos = model.instance_variable_get(:@cos)
    sin = model.instance_variable_get(:@sin)

    head_dim = @config.n_embd / @config.n_head
    # Rotary embeddings are precomputed with 10x over-allocation
    rotary_seq_len = @config.block_size * 10
    # cos/sin have half the head dimension (for rotary embedding computation)

    assert_equal [1, rotary_seq_len, 1, head_dim / 2], cos.size.to_a
    assert_equal [1, rotary_seq_len, 1, head_dim / 2], sin.size.to_a
  end

  def test_mlp_forward
    mlp = Nanochat::MLP.new(@config)
    input = Torch.randn(2, 10, 64) # [batch, seq, embd]
    output = mlp.forward(input)

    assert_equal [2, 10, 64], output.size.to_a
  end

  def test_mlp_is_torch_module = assert_kind_of(Torch::NN::Module, Nanochat::MLP.new(@config))

  def test_causal_self_attention_initialization
    attention = Nanochat::CausalSelfAttention.new(@config, 0)

    assert_kind_of Nanochat::CausalSelfAttention, attention
    assert_equal 0, attention.layer_idx
  end

  def test_causal_self_attention_forward
    attention = Nanochat::CausalSelfAttention.new(@config, 0)
    input = Torch.randn(2, 10, 64)
    head_dim = @config.n_embd / @config.n_head
    # cos/sin dimensions are half of head_dim for rotary embedding split
    cos = Torch.randn(1, 10, 1, head_dim / 2)
    sin = Torch.randn(1, 10, 1, head_dim / 2)
    cos_sin = [cos, sin]

    output = attention.forward(input, cos_sin, nil)

    assert_equal [2, 10, 64], output.size.to_a
  end

  def test_block_forward
    block = Nanochat::Block.new(@config, 0)
    input = Torch.randn(2, 10, 64)

    # Need cos_sin for RoPE
    head_dim = @config.n_embd / @config.n_head
    # cos/sin dimensions are half of head_dim for rotary embedding split
    cos = Torch.randn(1, 10, 1, head_dim / 2)
    sin = Torch.randn(1, 10, 1, head_dim / 2)
    cos_sin = [cos, sin]

    output = block.forward(input, cos_sin, nil)

    assert_equal [2, 10, 64], output.size.to_a
  end

  def test_block_is_torch_module = assert_kind_of(Torch::NN::Module, Nanochat::Block.new(@config, 0))

  def test_apply_rotary_emb_function
    input_tensor = Torch.randn(2, 4, 8, 16) # [batch, seq, heads, dim]
    # cos/sin should be half the dimension (for RoPE rotation)
    cos = Torch.randn(1, 4, 1, 8) # [1, seq, 1, dim/2]
    sin = Torch.randn(1, 4, 1, 8)

    output = Nanochat.apply_rotary_emb(input_tensor, cos, sin)

    assert_equal input_tensor.size.to_a, output.size.to_a
  end

  def test_apply_rotary_emb_requires_4d_tensor
    input_3d = Torch.randn(2, 8, 16)
    cos = Torch.randn(1, 8, 1, 8)
    sin = Torch.randn(1, 8, 1, 8)

    error = assert_raises(ArgumentError) do
      Nanochat.apply_rotary_emb(input_3d, cos, sin)
    end
    assert_match(/4D tensor/i, error.message)
  end

  def test_norm_function
    input_tensor = Torch.randn(2, 10, 64)
    output = Nanochat.norm(input_tensor)

    assert_equal input_tensor.size.to_a, output.size.to_a
  end
end
