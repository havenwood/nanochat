#!/usr/bin/env ruby
# frozen_string_literal: true

# Fine-tune a pre-trained nanochat model on custom text data
#
# Usage:
#   ruby examples/finetune.rb --checkpoint model.pt --data training.txt --output finetuned.pt
#
# Options:
#   -c, --checkpoint PATH   Pre-trained model checkpoint (default: ~/.cache/nanochat/model.pt)
#   -t, --tokenizer PATH    Tokenizer directory (default: ~/.cache/nanochat/tokenizer)
#   -d, --data PATH         Training data file (plain text)
#   -o, --output PATH       Output checkpoint path (default: finetuned.pt)
#   -e, --epochs N          Number of training epochs (default: 3)
#   -b, --batch-size N      Batch size (default: 4)
#   -s, --seq-length N      Sequence length (default: 512)
#   -l, --learning-rate LR  Learning rate (default: 5e-5)
#   -h, --help              Show this help message

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'
require 'optparse'

def parse_options
  options = {
    checkpoint: File.expand_path('~/.cache/nanochat/model.pt'),
    tokenizer_dir: File.expand_path('~/.cache/nanochat/tokenizer'),
    data: nil,
    output: 'finetuned.pt',
    epochs: 3,
    batch_size: 4,
    seq_length: 512,
    learning_rate: 5e-5
  }

  parser = OptionParser.new do |opts|
    opts.banner = 'Usage: finetune.rb [options]'
    opts.separator ''
    opts.separator 'Fine-tune a nanochat model on custom text data'
    opts.separator ''
    opts.separator 'Options:'

    opts.on('-c', '--checkpoint PATH', 'Pre-trained model checkpoint') do |path|
      options[:checkpoint] = File.expand_path(path)
    end

    opts.on('-t', '--tokenizer PATH', 'Tokenizer directory') do |path|
      options[:tokenizer_dir] = File.expand_path(path)
    end

    opts.on('-d', '--data PATH', 'Training data file (required)') do |path|
      options[:data] = File.expand_path(path)
    end

    opts.on('-o', '--output PATH', 'Output checkpoint path') do |path|
      options[:output] = File.expand_path(path)
    end

    opts.on('-e', '--epochs N', Integer, 'Number of epochs (default: 3)') do |n|
      options[:epochs] = n
    end

    opts.on('-b', '--batch-size N', Integer, 'Batch size (default: 4)') do |n|
      options[:batch_size] = n
    end

    opts.on('-s', '--seq-length N', Integer, 'Sequence length (default: 512)') do |n|
      options[:seq_length] = n
    end

    opts.on('-l', '--learning-rate LR', Float, 'Learning rate (default: 5e-5)') do |lr|
      options[:learning_rate] = lr
    end

    opts.on('-h', '--help', 'Show this help message') do
      puts opts
      exit
    end
  end

  parser.parse!

  if options[:data].nil?
    puts '❌ Error: Training data file is required (-d, --data PATH)'
    puts parser
    exit 1
  end

  options
end

def load_text_data(file_path, tokenizer, seq_length, batch_size)
  puts "📄 Loading training data from: #{file_path}"
  text = File.read(file_path)

  # Tokenize entire text
  tokens = tokenizer.encode(text)
  puts "  Total tokens: #{tokens.length}"

  # Split into sequences
  num_sequences = tokens.length / seq_length
  truncated_tokens = tokens[0, num_sequences * seq_length]

  # Reshape into sequences
  sequences = truncated_tokens.each_slice(seq_length).to_a
  puts "  Sequences: #{sequences.length}"

  # Create batches
  batches = sequences.each_slice(batch_size).to_a
  puts "  Batches: #{batches.length}"

  batches
end

def train_epoch(model, optimizer, batches, device, epoch, total_epochs)
  model.train
  total_loss = 0.0
  num_batches = batches.length

  batches.each_with_index do |batch, batch_idx|
    # Convert batch to tensor
    # Shape: [batch_size, seq_length]
    input_ids = Torch.tensor(batch, dtype: :long).to(device)

    # Create targets (shift input by 1)
    targets = input_ids[0..-1, 1..-1].contiguous
    input_ids = input_ids[0..-1, 0...-1].contiguous

    # Forward pass
    optimizer.zero_grad
    loss = model.call(input_ids, targets: targets)

    # Backward pass
    loss.backward

    # Clip gradients (prevent exploding gradients)
    Torch::NN::Utils.clip_grad_norm!(model.parameters, 1.0)

    # Optimizer step
    optimizer.step

    # Accumulate loss
    loss_value = loss.item
    total_loss += loss_value

    # Print progress
    next unless ((batch_idx + 1) % 10).zero? || batch_idx == num_batches - 1

    avg_loss = total_loss / (batch_idx + 1)
    progress = ((batch_idx + 1).to_f / num_batches * 100).round(1)
    puts "  Epoch #{epoch}/#{total_epochs} | Batch #{batch_idx + 1}/#{num_batches} (#{progress}%) | " \
         "Loss: #{loss_value.round(4)} | Avg Loss: #{avg_loss.round(4)}"
  end

  total_loss / num_batches
end

def main
  options = parse_options

  # Check files exist
  unless File.exist?(options[:checkpoint])
    puts "❌ Checkpoint not found: #{options[:checkpoint]}"
    exit 1
  end

  unless File.exist?(options[:data])
    puts "❌ Training data not found: #{options[:data]}"
    exit 1
  end

  tokenizer_file = File.join(options[:tokenizer_dir], 'tokenizer.json')
  unless File.exist?(tokenizer_file)
    puts "❌ Tokenizer not found: #{tokenizer_file}"
    exit 1
  end

  puts '🔄 Loading model and tokenizer...'

  # Load config from checkpoint
  checkpoint = Nanochat::CheckpointManager.load(options[:checkpoint])
  config = Nanochat::Config.from_checkpoint(checkpoint)

  # Load model
  model = Nanochat::GPT.from_checkpoint(options[:checkpoint], config)

  # Load tokenizer
  tokenizer = Nanochat::Tokenizer.from_directory(options[:tokenizer_dir])

  # Setup device
  device = Nanochat::Common.device
  model.to(device)

  puts "✅ Model loaded (device: #{device})"
  puts "  Config: #{config.n_layer} layers, #{config.n_embd} dim, #{config.vocab_size} vocab"
  puts

  # Load and prepare training data
  batches = load_text_data(
    options[:data],
    tokenizer,
    options[:seq_length],
    options[:batch_size]
  )
  puts

  # Setup optimizer
  puts '🔧 Setting up optimizer...'
  optimizer = Torch::Optim::AdamW.new(
    model.parameters,
    lr: options[:learning_rate],
    betas: [0.9, 0.95],
    weight_decay: 0.1
  )
  puts "  Optimizer: AdamW (lr=#{options[:learning_rate]})"
  puts

  # Training loop
  puts '🚀 Starting fine-tuning...'
  puts '=' * 70
  puts

  best_loss = Float::INFINITY

  options[:epochs].times do |epoch|
    epoch_num = epoch + 1
    puts "Epoch #{epoch_num}/#{options[:epochs]}"
    puts '-' * 70

    avg_loss = train_epoch(model, optimizer, batches, device, epoch_num, options[:epochs])

    puts "  ✅ Epoch #{epoch_num} complete | Average Loss: #{avg_loss.round(4)}"
    puts

    # Save best model
    next unless avg_loss < best_loss

    best_loss = avg_loss
    puts "  💾 New best model! Saving to #{options[:output]}"
    Nanochat::CheckpointManager.save(
      options[:output],
      model: model,
      config: config.to_h,
      epoch: epoch_num,
      loss: avg_loss,
      learning_rate: options[:learning_rate]
    )
    puts
  end

  puts '=' * 70
  puts '✅ Fine-tuning complete!'
  puts
  puts '📊 Training Summary:'
  puts "  Total epochs: #{options[:epochs]}"
  puts "  Best loss: #{best_loss.round(4)}"
  puts "  Model saved to: #{options[:output]}"
  puts
  puts '🎯 Next steps:'
  puts '  1. Test the fine-tuned model:'
  puts "     ruby examples/generate_text.rb -c #{options[:output]} \"Your prompt\""
  puts '  2. Chat with the fine-tuned model:'
  puts "     ruby examples/chat_cli.rb -c #{options[:output]}"
end

main if __FILE__ == $PROGRAM_NAME
