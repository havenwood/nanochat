# frozen_string_literal: true

# Nanochat: Minimal ChatGPT implementation in Ruby
# Ruby port of nanochat by Andrej Karpathy (https://github.com/karpathy/nanochat)
#
# This module provides transformer-based language model components including
# GPT architecture, attention mechanisms, and inference utilities.
module Nanochat
  # Helper Functions

  # RMSNorm - purely functional with no learnable parameters
  def self.norm(input)
    # Manual RMSNorm implementation since torch-rb doesn't have it built-in
    variance = input.pow(2).mean(-1, keepdim: true)
    input * Torch.rsqrt(variance + 1e-5)
  end

  # Apply Rotary Position Embeddings (RoPE)
  def self.apply_rotary_emb(input, cos, sin)
    raise ArgumentError, 'Expected 4D tensor for multihead attention' unless input.ndim == 4

    dim = input.shape[3] / 2
    # Split last dimension in half
    x1, x2 = input.split(dim, dim: -1)
    y1 = (x1 * cos) + (x2 * sin)
    y2 = (x1 * (-sin)) + (x2 * cos)
    out = Torch.cat([y1, y2], -1)
    out.to(dtype: input.dtype)
  end

  # Multi-Layer Perceptron (Feed-Forward Network)
  class MLP < Torch::NN::Module
    def initialize(config)
      super()
      @c_fc = Torch::NN::Linear.new(config.n_embd, 4 * config.n_embd, bias: false)
      @c_proj = Torch::NN::Linear.new(4 * config.n_embd, config.n_embd, bias: false)
    end

    def forward(input)
      x = @c_fc.call(input)
      x = Torch::NN::F.relu(x).square # ReLU^2 activation
      @c_proj.call(x)
    end
  end

  # Causal Self-Attention with Multi-Query Attention (MQA) support
  class CausalSelfAttention < Torch::NN::Module
    attr_reader :layer_idx

    def initialize(config, layer_idx)
      super()
      @layer_idx = layer_idx
      @n_head = config.n_head
      @n_kv_head = config.n_kv_head
      @n_embd = config.n_embd
      @head_dim = @n_embd / @n_head

      raise ArgumentError, 'n_embd must be divisible by n_head' unless (@n_embd % @n_head).zero?
      raise ArgumentError, 'Invalid MQA configuration' unless @n_kv_head <= @n_head && (@n_head % @n_kv_head).zero?

      @c_q = Torch::NN::Linear.new(@n_embd, @n_head * @head_dim, bias: false)
      @c_k = Torch::NN::Linear.new(@n_embd, @n_kv_head * @head_dim, bias: false)
      @c_v = Torch::NN::Linear.new(@n_embd, @n_kv_head * @head_dim, bias: false)
      @c_proj = Torch::NN::Linear.new(@n_embd, @n_embd, bias: false)
    end

    def forward(input, cos_sin, kv_cache)
      batch_size, seq_len, = input.size

      # Project to Q, K, V
      query = @c_q.call(input).view(batch_size, seq_len, @n_head, @head_dim)
      key = @c_k.call(input).view(batch_size, seq_len, @n_kv_head, @head_dim)
      value = @c_v.call(input).view(batch_size, seq_len, @n_kv_head, @head_dim)

      # Apply Rotary Embeddings and QK norm
      cos, sin = cos_sin
      query = Nanochat.apply_rotary_emb(query, cos, sin)
      key = Nanochat.apply_rotary_emb(key, cos, sin)
      query = Nanochat.norm(query)
      key = Nanochat.norm(key)

      # Transpose to make head the batch dimension: (B, T, H, D) -> (B, H, T, D)
      query = query.transpose(1, 2)
      key = key.transpose(1, 2)
      value = value.transpose(1, 2)

      # Apply KV cache if provided
      key, value = kv_cache.insert_kv(@layer_idx, key, value) if kv_cache

      tq = query.size(2)  # number of queries
      tk = key.size(2)    # number of keys (total in cache + current)

      # Expand key/value for Grouped Query Attention if needed
      if @n_head != @n_kv_head
        num_groups = @n_head / @n_kv_head
        key = key.repeat_interleave(num_groups, dim: 1)
        value = value.repeat_interleave(num_groups, dim: 1)
      end

      # Manual scaled dot-product attention (torch-rb doesn't support 4D multi-head attention)
      # Q, K, V are (B, H, T, D)
      scale = Math.sqrt(@head_dim)
      scores = Torch.matmul(query, key.transpose(-2, -1)) / scale # (B, H, Tq, Tk)

      # Apply causal mask
      if kv_cache.nil? || tq == tk
        # Training or full sequence: causal mask
        mask = Torch.ones([tq, tk], device: query.device).tril
        mask = Torch.zeros([tq, tk], device: query.device).masked_fill(mask.logical_not, -Float::INFINITY)
        scores += mask
      elsif tq == 1
        # Inference with single query: attend to all keys (no mask needed)
      else
        # Inference with multiple queries: prefix + causal
        mask = Torch.zeros([tq, tk], device: query.device)
        prefix_len = tk - tq
        if prefix_len.positive?
          # Full attention to prefix
          mask[0..-1, prefix_len..-1] = Torch.ones([tq, tq], device: query.device).tril.logical_not * -Float::INFINITY
        else
          mask = Torch.ones([tq, tk], device: query.device).tril.logical_not * -Float::INFINITY
        end
        scores += mask
      end

      # Softmax and apply to values
      attn_weights = Torch::NN::F.softmax(scores, dim: -1)
      y = Torch.matmul(attn_weights, value) # (B, H, Tq, D)

      # Re-assemble heads and project back
      y = y.transpose(1, 2).contiguous.view(batch_size, seq_len, -1)
      @c_proj.call(y)
    end
  end

  # Transformer Block
  class Block < Torch::NN::Module
    def initialize(config, layer_idx)
      super()
      @attn = CausalSelfAttention.new(config, layer_idx)
      @mlp = MLP.new(config)
    end

    def forward(input, cos_sin, kv_cache)
      # Attention with pre-norm and residual
      x = input + @attn.call(Nanochat.norm(input), cos_sin, kv_cache)
      # MLP with pre-norm and residual
      x + @mlp.call(Nanochat.norm(x))
    end
  end

  # Main GPT Model
  class GPT < Torch::NN::Module
    # Logit softcap value (prevents extreme logits)
    SOFTCAP = 15
    # Multiplier for pre-allocating rotary embedding cache
    ROTARY_CACHE_MULTIPLIER = 10

    attr_reader :config

    # Load a GPT model from a checkpoint file
    # @param path [String] Path to checkpoint file (.pt)
    # @param config [Config] Model configuration
    # @return [GPT] Loaded model instance
    def self.from_checkpoint(path, config)
      checkpoint = CheckpointManager.load(path)

      # Extract model state dict from checkpoint
      # Checkpoint structure: {'model' => state_dict, 'config' => {...}, ...}
      model_state = checkpoint['model'] || checkpoint[:model]
      raise ArgumentError, 'Checkpoint missing model state' unless model_state

      # Create model with given config
      model = new(config)

      # Load state dict into model
      # Note: torch-rb's load_state_dict requires string keys
      model.load_state_dict(model_state)

      model
    end

    def initialize(config)
      super()
      @config = config
      config.validate!

      # Transformer components
      @wte = Torch::NN::Embedding.new(config.vocab_size, config.n_embd)
      blocks = (0...config.n_layer).map { |idx| Block.new(config, idx) }
      @blocks = Torch::NN::ModuleList.new(blocks)
      @lm_head = Torch::NN::Linear.new(config.n_embd, config.vocab_size, bias: false)

      # Precompute rotary embeddings (over-allocate for efficiency)
      @rotary_seq_len = config.block_size * ROTARY_CACHE_MULTIPLIER
      head_dim = config.n_embd / config.n_head
      cos, sin = precompute_rotary_embeddings(@rotary_seq_len, head_dim)

      # Register as non-persistent buffers (not saved to checkpoint)
      register_buffer('cos', cos, persistent: false)
      register_buffer('sin', sin, persistent: false)
    end

    def forward(idx, targets: nil, kv_cache: nil)
      _, seq_len = idx.size

      # Get rotary embeddings for current sequence
      raise ArgumentError, "Sequence too long: #{seq_len} > #{@cos.size(1)}" if seq_len > @cos.size(1)
      raise ArgumentError, 'Device mismatch' unless idx.device == @cos.device

      # Offset rotary embeddings if using KV cache
      t0 = kv_cache.nil? ? 0 : kv_cache.pos
      # Slice temporal dimension: [1, total_len, 1, dim] -> [1, seq_len, 1, dim]
      cos = @cos.narrow(1, t0, seq_len)
      sin = @sin.narrow(1, t0, seq_len)
      cos_sin = [cos, sin]

      # Forward through transformer
      x = @wte.call(idx)
      x = Nanochat.norm(x)

      @blocks.each do |block|
        x = block.call(x, cos_sin, kv_cache)
      end

      x = Nanochat.norm(x)

      # Compute logits with softcap (prevents extreme values)
      logits = @lm_head.call(x)
      logits = SOFTCAP * Torch.tanh(logits / SOFTCAP)

      # Training: compute loss; Inference: return logits
      if targets
        logits = logits.float
        Torch::NN::F.cross_entropy(
          logits.view(-1, logits.size(-1)),
          targets.view(-1),
          ignore_index: -1
        )

      else
        logits
      end
    end

    # Generate tokens autoregressively
    def generate(tokens, max_tokens:, temperature: 1.0, top_k: nil, seed: 42)
      raise ArgumentError, 'tokens must be an Array' unless tokens.is_a?(Array)

      device = @wte.weight.device
      rng = temperature.positive? ? Torch::Generator.new(device).manual_seed(seed) : nil

      ids = Torch.tensor([tokens], dtype: :long, device: device)

      Torch.inference_mode do
        max_tokens.times do
          logits = forward(ids) # (B, T, vocab_size)
          logits = logits[0..-1, -1, 0..-1] # (B, vocab_size) - last position, preserve batch dim

          # Apply top-k filtering
          if top_k
            v, = Torch.topk(logits, [top_k, logits.size(-1)].min)
            logits[logits < v[0..-1, -1..-1]] = -Float::INFINITY
          end

          # Sample next token
          next_id = if temperature.positive?
                      probs = Torch::NN::F.softmax(logits / temperature, dim: -1)
                      Torch.multinomial(probs, num_samples: 1, generator: rng)
                    else
                      Torch.argmax(logits, dim: -1, keepdim: true)
                    end

          ids = Torch.cat([ids, next_id], dim: 1)
          yield next_id[0].item
        end
      end
    end

    private

    # Precompute rotary embeddings
    def precompute_rotary_embeddings(seq_len, head_dim, base: 10_000)
      device = @wte.weight.device

      # Frequency computation
      channel_range = Torch.arange(0, head_dim, 2, dtype: :float32, device: device)
      inv_freq = 1.0 / (base**(channel_range / head_dim))

      # Time steps
      time = Torch.arange(seq_len, dtype: :float32, device: device)

      # Calculate rotation frequencies
      freqs = Torch.outer(time, inv_freq)
      cos = freqs.cos.bfloat16
      sin = freqs.sin.bfloat16

      # Add batch and head dimensions for broadcasting: [seq_len, dim] -> [1, seq_len, 1, dim]
      cos = cos.unsqueeze(0).unsqueeze(2)
      sin = sin.unsqueeze(0).unsqueeze(2)

      [cos, sin]
    end
  end
end
