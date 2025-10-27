# frozen_string_literal: true

module Nanochat
  # GPT model configuration
  Config = Data.define(
    :vocab_size,
    :block_size,      # context length
    :n_embd,          # embedding dimension
    :n_head,          # query heads
    :n_kv_head,       # key/value heads (MQA)
    :n_layer          # transformer blocks
  ) do
    def self.default
      new(
        vocab_size: 50_304,
        block_size: 1024,
        n_embd: 768,
        n_head: 6,
        n_kv_head: 6,
        n_layer: 12
      )
    end

    def self.from_checkpoint(checkpoint)
      config_dict = checkpoint['config'] || checkpoint[:config]
      config_dict['block_size'] ||= config_dict['sequence_len'] if config_dict.is_a?(Hash)
      new(**config_dict.transform_keys(&:to_sym))
    end

    def validate!
      raise ArgumentError, "vocab_size (#{vocab_size}) must be positive" unless vocab_size.positive?
      raise ArgumentError, "block_size (#{block_size}) must be positive" unless block_size.positive?
      raise ArgumentError, "n_embd (#{n_embd}) must be positive" unless n_embd.positive?
      raise ArgumentError, "n_head (#{n_head}) must be positive" unless n_head.positive?
      raise ArgumentError, "n_kv_head (#{n_kv_head}) must be positive" unless n_kv_head.positive?
      raise ArgumentError, "n_layer (#{n_layer}) must be positive" unless n_layer.positive?
      raise ArgumentError, "n_embd (#{n_embd}) must be divisible by n_head (#{n_head})" unless (n_embd % n_head).zero?

      unless n_kv_head <= n_head
        raise ArgumentError,
              "Invalid MQA: n_kv_head (#{n_kv_head}) must be <= n_head (#{n_head})"
      end
      return if (n_head % n_kv_head).zero?

      raise ArgumentError,
            "Invalid MQA: n_head (#{n_head}) must be divisible by n_kv_head (#{n_kv_head})"
    end
  end
end
