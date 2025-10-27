# frozen_string_literal: true

# Nanochat Engine: Efficient inference with KV caching
# Ruby port of nanochat by Andrej Karpathy (https://github.com/karpathy/nanochat)

module Nanochat
  # KV Cache for efficient inference
  # Stores key/value pairs for each transformer layer
  class KVCache
    attr_reader :pos

    def initialize(batch_size, num_heads, seq_len, head_dim, num_layers)
      @kv_shape = [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
      @kv_cache = nil
      @pos = 0
    end

    def reset = @pos = 0

    def insert_kv(layer_idx, key, value)
      # Lazy initialize the cache with correct dtype/device
      @kv_cache = Torch.empty(@kv_shape, dtype: key.dtype, device: key.device) if @kv_cache.nil?

      # Insert new keys/values and return full cache so far
      _batch, _heads, t_add, _dim = key.size
      t0 = @pos
      t1 = @pos + t_add

      # Dynamically grow cache if needed
      if t1 > @kv_cache.size(4)
        t_needed = t1 + 1024 # Add buffer
        t_needed = (t_needed + 1023) & ~1023 # Round up to multiple of 1024
        current_shape = @kv_shape.dup
        current_shape[4] = t_needed
        @kv_cache = @kv_cache.resize(current_shape)
        @kv_shape = current_shape
      end

      # Insert k, v into cache
      @kv_cache[layer_idx, 0, 0..-1, 0..-1, t0...t1, 0..-1] = key
      @kv_cache[layer_idx, 1, 0..-1, 0..-1, t0...t1, 0..-1] = value

      # Return views of cached keys/values up to current position
      key_view = @kv_cache[layer_idx, 0, 0..-1, 0..-1, 0...t1, 0..-1]
      value_view = @kv_cache[layer_idx, 1, 0..-1, 0..-1, 0...t1, 0..-1]

      # Increment pos after last layer processes
      @pos = t1 if layer_idx == @kv_cache.size(0) - 1

      [key_view, value_view]
    end
  end

  # Inference engine for text generation
  class Engine
    def initialize(model:, tokenizer:, device: nil)
      @model = model
      @tokenizer = tokenizer
      @device = device || Common.device
      @model.to(@device)
      @model.eval # Set to evaluation mode
    end

    # Generate text from a prompt (non-streaming version)
    # Returns the complete generated text as a string
    def generate(prompt, max_tokens: 100, temperature: 1.0, top_k: nil, top_p: nil)
      tokens = []
      generate_stream(prompt, max_tokens:, temperature:, top_k:, top_p:) do |token_text, _token_id|
        tokens << token_text
      end
      tokens.join
    end

    # Generate text from a prompt with streaming (yields each token as generated)
    # Yields: token_text (String), token_id (Integer)
    # Can accept either a String prompt or Array of token IDs
    def generate_stream(prompt, max_tokens: 100, temperature: 1.0, top_k: nil, top_p: nil)
      # Handle both string prompts and token arrays
      tokens = prompt.is_a?(Array) ? prompt : @tokenizer.encode(prompt)
      return if tokens.empty? # Handle empty prompt

      # Initialize KV cache for efficient generation
      config = @model.config
      kv_cache = KVCache.new(
        1, # batch_size
        config.n_kv_head,
        tokens.length + max_tokens,
        config.n_embd / config.n_head,
        config.n_layer
      )

      # Convert tokens to tensor
      input_ids = Torch.tensor([tokens], dtype: :long).to(@device)
      generated_tokens = []

      Torch.no_grad do
        max_tokens.times do
          # Forward pass with KV cache
          logits = @model.call(input_ids, kv_cache:)
          next_token_logits = logits[0..-1, -1, 0..-1] # [batch, vocab_size]

          # Apply sampling
          next_token = sample(next_token_logits, temperature, top_k, top_p)
          token_id = next_token[0, 0].item

          # Stop on EOS token
          break if token_id == @tokenizer.eos_token_id

          # Decode and yield the token
          token_text = @tokenizer.decode([token_id])
          yield(token_text, token_id) if block_given?

          # Update input for next iteration (only the new token)
          input_ids = next_token.view(1, 1)
          generated_tokens << token_id
        end
      end

      generated_tokens
    end

    private

    # Sample next token from logits with temperature and top-k
    # logits shape: [batch, vocab_size], returns [batch, 1]
    def sample(logits, temperature, top_k, _top_p)
      # Greedy sampling (argmax) if temperature is 0
      return logits.argmax(-1, keepdim: true) if temperature.zero?

      # Top-k sampling
      if top_k
        k = [top_k, logits.size(-1)].min
        vals, idx = Torch.topk(logits, k, dim: -1)
        vals /= temperature
        probs = Torch::NN::F.softmax(vals, dim: -1)
        choice = Torch.multinomial(probs, num_samples: 1)
        return idx.gather(1, choice)
      end

      # Standard temperature sampling
      scaled_logits = logits / temperature
      probs = Torch::NN::F.softmax(scaled_logits, dim: -1)
      Torch.multinomial(probs, num_samples: 1)
    end
  end
end
