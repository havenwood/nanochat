# frozen_string_literal: true

require 'test_helper'

# Tests for Tokenizer using HuggingFace tokenizers gem
class TokenizerTest < Minitest::Test
  def setup
    # Create a minimal test tokenizer with a small vocabulary
    @tokenizer = create_test_tokenizer
  end

  def test_vocab_size
    assert_predicate @tokenizer.vocab_size, :positive?
    # Should be at least the number of special tokens
    assert_operator @tokenizer.vocab_size, :>=, Nanochat::Tokenizer::SPECIAL_TOKENS.length
  end

  def test_bos_token_id
    assert_kind_of Integer, @tokenizer.bos_token_id
    assert_operator @tokenizer.bos_token_id, :>=, 0
  end

  def test_eos_token_id
    # EOS should equal BOS in our implementation
    assert_equal @tokenizer.bos_token_id, @tokenizer.eos_token_id
  end

  def test_encode_simple_text
    tokens = @tokenizer.encode('hello')

    assert_instance_of Array, tokens
    assert tokens.all?(Integer)
    refute_empty tokens
  end

  def test_encode_and_decode_roundtrip
    text = 'Hello world!'
    tokens = @tokenizer.encode(text)
    decoded = @tokenizer.decode(tokens)

    # Decoded text should match original (may have whitespace differences)
    assert_equal text, decoded.strip
  end

  def test_encode_empty_string
    tokens = @tokenizer.encode('')

    assert_instance_of Array, tokens
    assert_empty tokens
  end

  def test_decode_empty_array
    text = @tokenizer.decode([])

    assert_instance_of String, text
    assert_equal '', text
  end

  def test_encode_with_prepend_id
    text = 'hello'
    prepend_id = 1
    tokens = @tokenizer.encode(text, prepend: prepend_id)

    assert_equal prepend_id, tokens.first
    assert_operator tokens.length, :>, @tokenizer.encode(text).length
  end

  def test_encode_with_append_id
    text = 'hello'
    append_id = 2
    tokens_plain = @tokenizer.encode(text)
    tokens_append = @tokenizer.encode(text, append: append_id)

    assert_equal append_id, tokens_append.last
    assert_equal tokens_append.length, tokens_plain.length + 1
  end

  def test_encode_array_of_strings
    texts = %w[hello world]
    results = @tokenizer.encode(texts)

    assert_instance_of Array, results
    assert_equal 2, results.length
    assert(results.all? { |tokens| tokens.is_a?(Array) })
  end

  def test_token_to_id_and_id_to_token
    token_id = @tokenizer.token_to_id('<|bos|>')

    if token_id
      assert_kind_of Integer, token_id
      token_string = @tokenizer.id_to_token(token_id)

      assert_equal '<|bos|>', token_string
    else
      # If special token doesn't exist, that's okay for basic tokenizer
      assert_nil token_id
    end
  end

  def test_from_directory_with_missing_file
    error = assert_raises(ArgumentError) do
      Nanochat::Tokenizer.from_directory('/nonexistent/directory')
    end

    assert_match(/not found/, error.message)
  end

  def test_special_tokens_present
    # All special tokens should be in vocabulary
    Nanochat::Tokenizer::SPECIAL_TOKENS.each do |token|
      token_id = @tokenizer.token_to_id(token)

      refute_nil token_id, "Special token #{token} not found in vocabulary"

      # Verify round-trip
      assert_equal token, @tokenizer.id_to_token(token_id)
    end
  end

  def test_encode_with_special_tokens
    # Test encoding with conversation-style special tokens
    text = 'hello'
    user_start_id = @tokenizer.token_to_id('<|user_start|>')
    user_end_id = @tokenizer.token_to_id('<|user_end|>')

    skip unless user_start_id && user_end_id

    tokens = @tokenizer.encode(text, prepend: user_start_id, append: user_end_id)

    assert_equal user_start_id, tokens.first
    assert_equal user_end_id, tokens.last
    assert_operator tokens.length, :>=, 3
  end

  def test_unicode_and_emoji
    # Test handling of unicode characters and emoji
    # Note: minimal test tokenizer may not preserve all unicode/emoji
    # This test validates the API works, even if output is lossy
    text = 'Hello 🌍 世界'
    tokens = @tokenizer.encode(text)
    decoded = @tokenizer.decode(tokens)

    # API should work without errors
    assert_instance_of Array, tokens
    assert_instance_of String, decoded
    refute_empty tokens
    refute_empty decoded
  end

  def test_long_text_encoding
    # Test with longer text from training corpus to ensure no issues with long sequences
    # Use text that was in the training data so vocabulary can handle it
    long_text = 'hello world this is a test ' * 20
    tokens = @tokenizer.encode(long_text)
    decoded = @tokenizer.decode(tokens)

    # Should produce a long sequence of tokens
    assert_operator tokens.length, :>, 50
    # Decoding should produce reasonable output (may not be exact due to BPE merges)
    assert_instance_of String, decoded
    assert_operator decoded.length, :>, 50
    # Should contain at least some of the original words
    assert_match(/hello|world|test/i, decoded)
  end

  def test_whitespace_handling
    # Test various whitespace scenarios with text from training corpus
    texts = [
      '  hello world',
      'hello world  ',
      'hello    world',
      "\thello\tworld\t",
      "hello\nworld"
    ]

    texts.each do |text|
      tokens = @tokenizer.encode(text)
      decoded = @tokenizer.decode(tokens)

      # API should work without errors
      refute_empty tokens
      refute_empty decoded
      # Should contain recognizable words from training data
      assert_match(/hello|world/i, decoded)
    end
  end

  def test_batch_decode
    # Test decoding multiple sequences (if supported)
    text1 = 'hello'
    text2 = 'world'
    tokens1 = @tokenizer.encode(text1)
    tokens2 = @tokenizer.encode(text2)

    # Individual decoding should work
    assert_equal text1, @tokenizer.decode(tokens1).strip
    assert_equal text2, @tokenizer.decode(tokens2).strip
  end

  private

  # Create a minimal test tokenizer with byte-level BPE
  def create_test_tokenizer
    require 'tempfile'

    # Create a simple BPE tokenizer with byte-level pre-tokenization
    underlying = Tokenizers::Tokenizer.new(
      Tokenizers::Models::BPE.new(unk_token: '<|bos|>')
    )

    # Add byte-level pre-tokenization
    underlying.pre_tokenizer = Tokenizers::PreTokenizers::ByteLevel.new(
      add_prefix_space: false
    )

    # Add byte-level decoder
    underlying.decoder = Tokenizers::Decoders::ByteLevel.new

    # Train the tokenizer with a small corpus
    trainer = Tokenizers::Trainers::BpeTrainer.new(
      vocab_size: 300,
      special_tokens: Nanochat::Tokenizer::SPECIAL_TOKENS,
      show_progress: false,
      min_frequency: 1
    )

    # Create temporary training file
    Tempfile.create(['train', '.txt']) do |file|
      # Write diverse text to build vocabulary
      file.write('hello world this is a test Hello world! ' * 50)
      file.flush
      underlying.train([file.path], trainer)
    end

    Nanochat::Tokenizer.new(underlying)
  end
end
