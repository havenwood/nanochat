# frozen_string_literal: true

# Nanochat Tokenizer: BPE tokenization using HuggingFace tokenizers
# Ruby port of nanochat by Andrej Karpathy (https://github.com/karpathy/nanochat)

require 'tokenizers'

module Nanochat
  # BPE Tokenizer using HuggingFace tokenizers gem for Ruby
  # Compatible with Python nanochat's HuggingFaceTokenizer (tokenizer.json format)
  #
  # This implementation wraps the tokenizers gem and provides an API compatible
  # with Python nanochat. For best compatibility, train tokenizers in Python using
  # HuggingFaceTokenizer and save to tokenizer.json format.
  class Tokenizer
    # Special tokens used by nanochat (from Python implementation)
    SPECIAL_TOKENS = [
      '<|bos|>',
      '<|user_start|>', '<|user_end|>',
      '<|assistant_start|>', '<|assistant_end|>',
      '<|python_start|>', '<|python_end|>',
      '<|output_start|>', '<|output_end|>'
    ].freeze

    attr_reader :tokenizer, :bos_token_id

    # Initialize with a tokenizers gem Tokenizer instance
    # @param tokenizer [Tokenizers::Tokenizer] The underlying tokenizer
    def initialize(tokenizer)
      @tokenizer = tokenizer
      @bos_token_id = token_to_id('<|bos|>') || 0
    end

    # Load a pretrained tokenizer from HuggingFace Hub
    # @param identifier [String] Model identifier (e.g., "bert-base-cased")
    # @return [Tokenizer] New tokenizer instance
    def self.from_pretrained(identifier)
      tokenizer = Tokenizers.from_pretrained(identifier)
      new(tokenizer)
    end

    # Load tokenizer from a tokenizer.json file
    # @param path [String] Path to tokenizer.json file
    # @return [Tokenizer] New tokenizer instance
    def self.from_file(path)
      tokenizer = Tokenizers.from_file(path)
      new(tokenizer)
    end

    # Load tokenizer from a directory containing tokenizer.json
    # @param directory [String] Directory path
    # @return [Tokenizer] New tokenizer instance
    def self.from_directory(directory)
      tokenizer_path = File.join(directory, 'tokenizer.json')
      raise ArgumentError, "Tokenizer file not found: #{tokenizer_path}" unless File.exist?(tokenizer_path)

      from_file(tokenizer_path)
    end

    # Get the vocabulary size
    # @return [Integer] Number of tokens in vocabulary
    def vocab_size = @tokenizer.vocab_size

    # Get EOS token ID (for compatibility with Python nanochat)
    # In this implementation, EOS is the same as BOS
    # @return [Integer] End-of-sequence token ID
    def eos_token_id = @bos_token_id

    # Encode text to token IDs
    # @param text [String, Array<String>] Text to encode (single string or array)
    # @param prepend [String, Integer, nil] Token or ID to prepend
    # @param append [String, Integer, nil] Token or ID to append
    # @return [Array<Integer>, Array<Array<Integer>>] Token IDs
    def encode(text, prepend: nil, append: nil)
      # Convert prepend/append to IDs if they're strings
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

    # Decode token IDs to text
    # @param ids [Array<Integer>] Token IDs to decode
    # @return [String] Decoded text
    def decode(ids)
      return '' if ids.empty?

      @tokenizer.decode(ids)
    end

    # Get token ID for a specific token string
    # @param token [String] Token string
    # @return [Integer, nil] Token ID or nil if not found
    def token_to_id(token) = @tokenizer.token_to_id(token)

    # Get token string for a specific token ID
    # @param token_id [Integer] Token ID
    # @return [String, nil] Token string or nil if not found
    def id_to_token(token_id) = @tokenizer.id_to_token(token_id)

    # Save tokenizer to a directory
    # @param directory [String] Directory path
    def save(directory)
      require 'fileutils'
      FileUtils.mkdir_p(directory)
      tokenizer_path = File.join(directory, 'tokenizer.json')
      @tokenizer.save(tokenizer_path)
      puts "Saved tokenizer to #{tokenizer_path}"
    end

    # Train a new BPE tokenizer from text files
    # @param files [Array<String>] Array of file paths
    # @param vocab_size [Integer] Vocabulary size
    # @param special_tokens [Array<String>] Special tokens to add
    # @return [Tokenizer] New trained tokenizer
    def self.train_from_files(files, vocab_size:, special_tokens: SPECIAL_TOKENS)
      tokenizer = Tokenizers::Tokenizer.new(Tokenizers::Models::BPE.new(
                                              unk_token: special_tokens.first
                                            ))

      # Set pre-tokenizer (GPT-4 style splitting)
      tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::ByteLevel.new(
        add_prefix_space: false,
        use_regex: true
      )

      # Set decoder
      tokenizer.decoder = Tokenizers::Decoders::ByteLevel.new

      # Train
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

    # Encode a single string (internal helper)
    # @param text [String] Text to encode
    # @param prepend_id [Integer, nil] Token ID to prepend
    # @param append_id [Integer, nil] Token ID to append
    # @return [Array<Integer>] Token IDs
    def encode_single(text, prepend_id, append_id)
      # Encode without adding special tokens
      encoding = @tokenizer.encode(text)
      ids = encoding.ids

      # Add prepend/append tokens
      ids.unshift(prepend_id) if prepend_id
      ids << append_id if append_id

      ids
    end
  end
end
