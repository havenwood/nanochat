#!/usr/bin/env ruby
# frozen_string_literal: true

# Interactive chat CLI for nanochat
# Mirrors python-nanochat/scripts/chat_cli.py functionality
#
# Usage:
#   ruby examples/chat_cli.rb [options]
#
# Options:
#   -c, --checkpoint PATH   Path to model checkpoint (default: ~/.cache/nanochat/model.pt)
#   -t, --tokenizer PATH    Path to tokenizer directory (default: ~/.cache/nanochat/tokenizer)
#   -T, --temperature TEMP  Temperature for generation (default: 0.6)
#   -k, --top-k K          Top-k sampling parameter (default: 50)
#   -m, --max-tokens N     Max tokens to generate (default: 256)
#   -p, --prompt TEXT      Single prompt mode (generate once and exit)
#   -h, --help             Show this help message

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'
require 'optparse'

def parse_options
  options = {
    checkpoint: File.expand_path('~/.cache/nanochat/model.pt'),
    tokenizer_dir: File.expand_path('~/.cache/nanochat/tokenizer'),
    temperature: 0.6,
    top_k: 50,
    max_tokens: 256,
    prompt: nil
  }

  parser = OptionParser.new do |opts|
    opts.banner = 'Usage: chat_cli.rb [options]'
    opts.separator ''
    opts.separator 'Interactive chat with nanochat model'
    opts.separator ''
    opts.separator 'Options:'

    opts.on('-c', '--checkpoint PATH', 'Path to model checkpoint') do |path|
      options[:checkpoint] = File.expand_path(path)
    end

    opts.on('-t', '--tokenizer PATH', 'Path to tokenizer directory') do |path|
      options[:tokenizer_dir] = File.expand_path(path)
    end

    opts.on('-T', '--temperature TEMP', Float, 'Temperature (default: 0.6)') do |temp|
      options[:temperature] = temp
    end

    opts.on('-k', '--top-k K', Integer, 'Top-k sampling (default: 50)') do |k|
      options[:top_k] = k
    end

    opts.on('-m', '--max-tokens N', Integer, 'Max tokens to generate (default: 256)') do |n|
      options[:max_tokens] = n
    end

    opts.on('-p', '--prompt TEXT', 'Single prompt mode (generate once and exit)') do |text|
      options[:prompt] = text
    end

    opts.on('-h', '--help', 'Show this help message') do
      puts opts
      exit
    end
  end

  parser.parse!
  options
end

def main
  options = parse_options

  # Check if checkpoint exists
  unless File.exist?(options[:checkpoint])
    puts "❌ Checkpoint not found: #{options[:checkpoint]}"
    puts "\nTo train a demo model (~30 mins):"
    puts '  bash bin/speedrun.sh'
    exit 1
  end

  # Check if tokenizer exists
  tokenizer_file = File.join(options[:tokenizer_dir], 'tokenizer.json')
  unless File.exist?(tokenizer_file)
    puts "❌ Tokenizer not found: #{tokenizer_file}"
    puts "\nEnsure your Python training uses HuggingFaceTokenizer"
    exit 1
  end

  # Load model and tokenizer
  puts '🔄 Loading model and tokenizer...'
  config = Nanochat::Config.from_checkpoint(
    Nanochat::CheckpointManager.load(options[:checkpoint])
  )
  model = Nanochat::GPT.from_checkpoint(options[:checkpoint], config)
  tokenizer = Nanochat::Tokenizer.from_directory(options[:tokenizer_dir])
  device = Nanochat::Common.device

  puts "✅ Model loaded (device: #{device})"
  puts

  # Create engine
  engine = Nanochat::Engine.new(model:, tokenizer:, device:)

  # Special tokens for conversation state machine
  bos = tokenizer.bos_token_id
  user_start = tokenizer.token_to_id('<|user_start|>')
  user_end = tokenizer.token_to_id('<|user_end|>')
  assistant_start = tokenizer.token_to_id('<|assistant_start|>')
  assistant_end = tokenizer.token_to_id('<|assistant_end|>')

  # Print welcome message
  puts 'NanoChat Interactive Mode'
  puts '-' * 50
  puts "Type 'quit' or 'exit' to end the conversation"
  puts "Type 'clear' to start a new conversation"
  puts '-' * 50

  # Initialize conversation with BOS token
  conversation_tokens = [bos]

  loop do
    # Get user input
    if options[:prompt]
      user_input = options[:prompt]
    else
      print "\nUser: "
      user_input = $stdin.gets&.chomp
      break if user_input.nil?
    end

    # Handle special commands
    if %w[quit exit].include?(user_input&.downcase)
      puts 'Goodbye!'
      break
    end

    if user_input&.downcase == 'clear'
      conversation_tokens = [bos]
      puts 'Conversation cleared.'
      next
    end

    next if user_input.nil? || user_input.strip.empty?

    # Add user message to conversation
    conversation_tokens << user_start
    conversation_tokens.concat(tokenizer.encode(user_input))
    conversation_tokens << user_end

    # Start assistant response
    conversation_tokens << assistant_start
    response_tokens = []

    print "\nAssistant: "
    $stdout.flush

    # Generate with streaming
    engine.generate_stream(
      conversation_tokens,
      max_tokens: options[:max_tokens],
      temperature: options[:temperature],
      top_k: options[:top_k]
    ) do |token_text, token_id|
      # Stop on assistant_end token
      break if token_id == assistant_end

      print token_text
      $stdout.flush
      response_tokens << token_id
    end

    puts # Newline after response

    # Ensure assistant_end is at the end
    response_tokens << assistant_end unless response_tokens.last == assistant_end
    conversation_tokens.concat(response_tokens)

    # In single-prompt mode, exit after one response
    break if options[:prompt]
  end
end

main if __FILE__ == $PROGRAM_NAME
