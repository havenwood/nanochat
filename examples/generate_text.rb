#!/usr/bin/env ruby
# frozen_string_literal: true

# Simple text generation example
# Mirrors basic generation functionality from python-nanochat
#
# Usage:
#   ruby examples/generate_text.rb [options] "Your prompt here"
#
# Options:
#   -c, --checkpoint PATH   Path to model checkpoint (default: ~/.cache/nanochat/model.pt)
#   -t, --tokenizer PATH    Path to tokenizer directory (default: ~/.cache/nanochat/tokenizer)
#   -T, --temperature TEMP  Temperature for generation (default: 0.8)
#   -k, --top-k K          Top-k sampling parameter (default: 50)
#   -m, --max-tokens N     Max tokens to generate (default: 200)
#   -s, --stream           Stream output token by token
#   -h, --help             Show this help message

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'
require 'optparse'

def parse_options
  options = {
    checkpoint: File.expand_path('~/.cache/nanochat/model.pt'),
    tokenizer_dir: File.expand_path('~/.cache/nanochat/tokenizer'),
    temperature: 0.8,
    top_k: 50,
    max_tokens: 200,
    stream: false,
    prompt: nil
  }

  parser = OptionParser.new do |opts|
    opts.banner = 'Usage: generate_text.rb [options] "prompt text"'
    opts.separator ''
    opts.separator 'Generate text from a prompt using nanochat model'
    opts.separator ''
    opts.separator 'Options:'

    opts.on('-c', '--checkpoint PATH', 'Path to model checkpoint') do |path|
      options[:checkpoint] = File.expand_path(path)
    end

    opts.on('-t', '--tokenizer PATH', 'Path to tokenizer directory') do |path|
      options[:tokenizer_dir] = File.expand_path(path)
    end

    opts.on('-T', '--temperature TEMP', Float, 'Temperature (default: 0.8)') do |temp|
      options[:temperature] = temp
    end

    opts.on('-k', '--top-k K', Integer, 'Top-k sampling (default: 50)') do |k|
      options[:top_k] = k
    end

    opts.on('-m', '--max-tokens N', Integer, 'Max tokens to generate (default: 200)') do |n|
      options[:max_tokens] = n
    end

    opts.on('-s', '--stream', 'Stream output token by token') do
      options[:stream] = true
    end

    opts.on('-h', '--help', 'Show this help message') do
      puts opts
      exit
    end
  end

  parser.parse!
  options[:prompt] = ARGV.join(' ') unless ARGV.empty?
  options
end

def main
  options = parse_options

  # Get prompt (from args or default)
  prompt = options[:prompt] || 'Once upon a time'

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
  puts "Generating text from prompt: \"#{prompt}\""
  puts '=' * 60
  puts

  # Create engine
  engine = Nanochat::Engine.new(model:, tokenizer:, device:)

  # Generate text
  if options[:stream]
    # Streaming mode
    print prompt
    engine.generate_stream(
      prompt,
      max_tokens: options[:max_tokens],
      temperature: options[:temperature],
      top_k: options[:top_k]
    ) do |token_text, _token_id|
      print token_text
      $stdout.flush
    end
    puts
  else
    # Non-streaming mode
    text = engine.generate(
      prompt,
      max_tokens: options[:max_tokens],
      temperature: options[:temperature],
      top_k: options[:top_k]
    )
    puts text
  end
end

main if __FILE__ == $PROGRAM_NAME
