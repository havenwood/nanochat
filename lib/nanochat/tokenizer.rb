# frozen_string_literal: true

# Nanochat Tokenizer: BPE tokenization using HuggingFace tokenizers
# Ruby port of nanochat by Andrej Karpathy (https://github.com/karpathy/nanochat)

require 'tokenizers'

module Nanochat
  # BPE tokenizer wrapping HuggingFace's tokenizers gem.
  # Compatible with Python nanochat's tokenizer.json format.
  class Tokenizer
    SPECIAL_TOKENS = [
      '<|bos|>',
      '<|user_start|>', '<|user_end|>',
      '<|assistant_start|>', '<|assistant_end|>',
      '<|python_start|>', '<|python_end|>',
      '<|output_start|>', '<|output_end|>'
    ].freeze

    attr_reader :tokenizer, :bos_token_id

    def initialize(tokenizer)
      @tokenizer = tokenizer
      @bos_token_id = token_to_id('<|bos|>') || 0
    end

    def self.from_pretrained(identifier)
      tokenizer = Tokenizers.from_pretrained(identifier)
      new(tokenizer)
    end

    def self.from_file(path)
      tokenizer = Tokenizers.from_file(path)
      new(tokenizer)
    end

    def self.from_directory(directory)
      tokenizer_path = File.join(directory, 'tokenizer.json')
      raise ArgumentError, "Tokenizer file not found: #{tokenizer_path}" unless File.exist?(tokenizer_path)

      from_file(tokenizer_path)
    end

    def vocab_size = @tokenizer.vocab_size

    # EOS is the same as BOS for compatibility
    def eos_token_id = @bos_token_id

    # Encode text to token IDs. Accepts strings or arrays of strings.
    # Pass prepend/append as token strings or IDs to add special tokens.
    def encode(text, prepend: nil, append: nil)
      prepend_id = prepend.is_a?(String) ? token_to_id(prepend) : prepend
      append_id = append.is_a?(String) ? token_to_id(append) : append

      case text
      when String
        encode_single(text, prepend_id, append_id)
      when Array
        text.map { |str| encode_single(str, prepend_id, append_id) }
      else
        raise ArgumentError, "Invalid input type: #{text.class}"
      end
    end

    def decode(ids)
      return '' if ids.empty?

      @tokenizer.decode(ids)
    end

    def token_to_id(token) = @tokenizer.token_to_id(token)

    def id_to_token(token_id) = @tokenizer.id_to_token(token_id)

    def save(directory)
      require 'fileutils'
      FileUtils.mkdir_p(directory)
      tokenizer_path = File.join(directory, 'tokenizer.json')
      @tokenizer.save(tokenizer_path)
      puts "Saved tokenizer to #{tokenizer_path}"
    end

    # Train a new BPE tokenizer from text files
    def self.train_from_files(files, vocab_size:, special_tokens: SPECIAL_TOKENS)
      tokenizer = Tokenizers::Tokenizer.new(Tokenizers::Models::BPE.new(
                                              unk_token: special_tokens.first
                                            ))

      tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::ByteLevel.new(
        add_prefix_space: false,
        use_regex: true
      )

      tokenizer.decoder = Tokenizers::Decoders::ByteLevel.new

      trainer = Tokenizers::Trainers::BpeTrainer.new(
        vocab_size:,
        special_tokens:,
        show_progress: true,
        min_frequency: 0
      )

      tokenizer.train(files, trainer)
      new(tokenizer)
    end

    private

    def encode_single(text, prepend_id, append_id)
      encoding = @tokenizer.encode(text)
      ids = encoding.ids

      ids.unshift(prepend_id) if prepend_id
      ids << append_id if append_id

      ids
    end
  end
end
