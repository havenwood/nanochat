# frozen_string_literal: true

# Nanochat Engine: Efficient inference with KV caching
# Ruby port of nanochat by Andrej Karpathy (https://github.com/karpathy/nanochat)

module Nanochat
  # KV cache for efficient inference
  class KVCache
    attr_reader :pos

    def initialize(batch_size, num_heads, seq_len, head_dim, num_layers)
      @kv_shape = [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
      @kv_cache = nil
      @pos = 0
    end

    def reset = @pos = 0

    def insert_kv(layer_idx, key, value)
      @kv_cache = Torch.empty(@kv_shape, dtype: key.dtype, device: key.device) if @kv_cache.nil?

      _batch, _heads, t_add, _dim = key.size
      t0 = @pos
      t1 = @pos + t_add

      if t1 > @kv_cache.size(4)
        t_needed = t1 + 1024
        t_needed = (t_needed + 1023) & ~1023
        current_shape = @kv_shape.dup
        current_shape[4] = t_needed
        @kv_cache = @kv_cache.resize(current_shape)
        @kv_shape = current_shape
      end

      @kv_cache[layer_idx, 0, 0..-1, 0..-1, t0...t1, 0..-1] = key
      @kv_cache[layer_idx, 1, 0..-1, 0..-1, t0...t1, 0..-1] = value

      key_view = @kv_cache[layer_idx, 0, 0..-1, 0..-1, 0...t1, 0..-1]
      value_view = @kv_cache[layer_idx, 1, 0..-1, 0..-1, 0...t1, 0..-1]

      @pos = t1 if layer_idx == @kv_cache.size(0) - 1

      [key_view, value_view]
    end
  end

  # Text generation engine
  class Engine
    def initialize(model:, tokenizer:, device: nil)
      @model = model
      @tokenizer = tokenizer
      @device = device || Common.device
      @model.to(@device)
      @model.eval
    end

    def generate(prompt, max_tokens: 100, temperature: 1.0, top_k: nil, top_p: nil)
      tokens = []
      generate_stream(prompt, max_tokens:, temperature:, top_k:, top_p:) do |token_text, _token_id|
        tokens << token_text
      end
      tokens.join
    end

    # Generate text with streaming. Yields token_text (String), token_id (Integer).
    # Accepts string prompts or token arrays.
    def generate_stream(prompt, max_tokens: 100, temperature: 1.0, top_k: nil, top_p: nil)
      tokens = prompt.is_a?(Array) ? prompt : @tokenizer.encode(prompt)
      return if tokens.empty?

      config = @model.config
      kv_cache = KVCache.new(
        1,
        config.n_kv_head,
        tokens.length + max_tokens,
        config.n_embd / config.n_head,
        config.n_layer
      )

      input_ids = Torch.tensor([tokens], dtype: :long).to(@device)
      generated_tokens = []

      Torch.no_grad do
        max_tokens.times do
          logits = @model.call(input_ids, kv_cache:)
          next_token_logits = logits[0..-1, -1, 0..-1]

          next_token = sample(next_token_logits, temperature, top_k, top_p)
          token_id = next_token[0, 0].item

          break if token_id == @tokenizer.eos_token_id

          token_text = @tokenizer.decode([token_id])
          yield(token_text, token_id) if block_given?

          input_ids = next_token.view(1, 1)
          generated_tokens << token_id
        end
      end

      generated_tokens
    end

    private

    def sample(logits, temperature, top_k, top_p)
      return logits.argmax(-1, keepdim: true) if temperature.zero?

      if top_k
        k = [top_k, logits.size(-1)].min
        vals, idx = Torch.topk(logits, k, dim: -1)
        vals /= temperature
        probs = Torch::NN::F.softmax(vals, dim: -1)
        choice = Torch.multinomial(probs, num_samples: 1)
        return idx.gather(1, choice)
      end

      # Top-p (nucleus) sampling
      if top_p && top_p < 1.0
        scaled_logits = logits / temperature
        probs = Torch::NN::F.softmax(scaled_logits, dim: -1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = probs.sort(dim: -1, descending: true)

        # Compute cumulative probabilities
        cumulative_probs = sorted_probs.cumsum(dim: -1)

        # Remove tokens with cumulative probability above threshold
        # Keep at least one token (the highest probability one)
        sorted_indices_to_remove = Torch.gt(cumulative_probs, top_p)
        sorted_indices_to_remove[0..-1, 0] = false

        # Zero out probabilities for removed tokens
        sorted_probs[sorted_indices_to_remove] = 0.0

        # Renormalize probabilities
        sorted_probs /= sorted_probs.sum(dim: -1, keepdim: true)

        # Sample from filtered distribution
        choice = Torch.multinomial(sorted_probs, num_samples: 1)

        # Map back to original vocabulary indices
        return sorted_indices.gather(1, choice)
      end

      scaled_logits = logits / temperature
      probs = Torch::NN::F.softmax(scaled_logits, dim: -1)
      Torch.multinomial(probs, num_samples: 1)
    end
  end
end
