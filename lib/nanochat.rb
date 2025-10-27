# frozen_string_literal: true

require 'torch'
require_relative 'nanochat/version'
require_relative 'nanochat/common'
require_relative 'nanochat/config'
require_relative 'nanochat/tokenizer'
require_relative 'nanochat/gpt'
require_relative 'nanochat/checkpoint_manager'
require_relative 'nanochat/engine'

# Nanochat: Minimal ChatGPT implementation in Ruby
# Port of https://github.com/karpathy/nanochat
module Nanochat
  class Error < StandardError; end

  # Convenience method to load a model and create an engine
  def self.load_model(checkpoint_path, tokenizer: nil)
    checkpoint = CheckpointManager.load(checkpoint_path)
    config = Config.from_checkpoint(checkpoint)
    model = GPT.new(config)
    model.load_state_dict(checkpoint[:model] || checkpoint['model'])

    tokenizer ||= Tokenizer.new(vocab_size: config.vocab_size)

    Engine.new(model: model, tokenizer: tokenizer)
  end
end
