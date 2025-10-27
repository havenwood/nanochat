#!/usr/bin/env ruby
# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'
require 'benchmark/ips'
require 'optparse'

# Benchmark suite for Nanochat Ruby
# Measures performance of key operations to match Python nanochat benchmarking
#
# Usage:
#   ruby examples/benchmark.rb
#   ruby examples/benchmark.rb --model path/to/model.pt --tokenizer path/to/tokenizer

module NanochatBenchmark
  class Runner
    def initialize(model_path: nil, tokenizer_path: nil)
      @model_path = model_path || File.expand_path('~/.cache/nanochat/model.pt')
      @tokenizer_path = tokenizer_path || File.expand_path('~/.cache/nanochat/tokenizer')
      @device = Nanochat::Common.device

      puts 'Nanochat Benchmark Suite'
      puts '=' * 80
      puts "Ruby version: #{RUBY_VERSION}"
      puts "Device: #{@device}"
      puts "Model: #{@model_path}"
      puts "Tokenizer: #{@tokenizer_path}"
      puts '=' * 80
      puts
    end

    def run_all
      if can_load_model?
        run_with_real_model
      else
        run_with_mock_model
      end
    end

    private

    def can_load_model?
      File.exist?(@model_path) && File.exist?(File.join(@tokenizer_path, 'tokenizer.json'))
    end

    def run_with_real_model
      puts 'Loading real model...'

      # Load model and tokenizer
      checkpoint = Nanochat::CheckpointManager.load(@model_path)
      config = Nanochat::Config.from_checkpoint(checkpoint)
      model = Nanochat::GPT.from_checkpoint(@model_path, config)
      tokenizer = Nanochat::Tokenizer.from_directory(@tokenizer_path)

      puts "Model loaded: #{config.n_layer} layers, #{config.n_embd} embedding dim"
      puts

      benchmark_tokenizer(tokenizer)
      benchmark_model(model, tokenizer, config)
      benchmark_generation(model, tokenizer)
    end

    def run_with_mock_model
      puts "⚠️  No trained model found at #{@model_path}"
      puts 'Running benchmarks with mock model (architecture validation only)'
      puts

      # Create tiny model for benchmarking
      config = Nanochat::Config.new(
        vocab_size: 512,
        block_size: 128,
        n_embd: 64,
        n_head: 4,
        n_kv_head: 2,
        n_layer: 2
      )

      model = Nanochat::GPT.new(config)
      model.to(@device)

      benchmark_model_architecture(model, config)
    end

    def benchmark_tokenizer(tokenizer)
      puts '📊 Tokenizer Benchmarks'
      puts '-' * 80

      # Test texts of varying lengths
      short_text = 'Hello, world!'
      medium_text = 'The quick brown fox jumps over the lazy dog. ' * 10
      long_text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 100

      # Encode benchmarks
      puts "\nEncode performance:"

      short_ids = nil
      Benchmark.ips do |x|
        x.config(time: 5, warmup: 2)

        x.report('encode (short, 13 chars)') do
          short_ids = tokenizer.encode(short_text)
        end

        x.report('encode (medium, ~460 chars)') do
          tokenizer.encode(medium_text)
        end

        x.report('encode (long, ~5700 chars)') do
          tokenizer.encode(long_text)
        end

        x.compare!
      end

      # Decode benchmarks
      puts "\nDecode performance:"

      medium_ids = tokenizer.encode(medium_text)
      long_ids = tokenizer.encode(long_text)

      Benchmark.ips do |x|
        x.config(time: 5, warmup: 2)

        x.report('decode (short, ~3 tokens)') do
          tokenizer.decode(short_ids) if short_ids
        end

        x.report('decode (medium, ~100 tokens)') do
          tokenizer.decode(medium_ids)
        end

        x.report('decode (long, ~1200 tokens)') do
          tokenizer.decode(long_ids)
        end

        x.compare!
      end

      puts
    end

    def benchmark_model(model, _tokenizer, config)
      puts '📊 Model Benchmarks'
      puts '-' * 80

      # Create input tensors
      batch_sizes = [1, 4]
      seq_lengths = [16, 64, 128]

      puts "\nForward pass latency:"

      Benchmark.ips do |x|
        x.config(time: 5, warmup: 2)

        batch_sizes.each do |batch_size|
          seq_lengths.each do |seq_len|
            next if seq_len > config.block_size

            input_ids = Torch.randint(0, config.vocab_size, [batch_size, seq_len], dtype: :long).to(@device)

            x.report("forward (B=#{batch_size}, T=#{seq_len})") do
              Torch.no_grad do
                model.call(input_ids)
              end
            end
          end
        end

        x.compare!
      end

      puts
    end

    def benchmark_model_architecture(model, config)
      puts '📊 Model Architecture Benchmarks'
      puts '-' * 80

      seq_lengths = [16, 64, 128]

      puts "\nForward pass latency (mock model):"

      Benchmark.ips do |x|
        x.config(time: 5, warmup: 2)

        seq_lengths.each do |seq_len|
          next if seq_len > config.block_size

          input_ids = Torch.randint(0, config.vocab_size, [1, seq_len], dtype: :long).to(@device)

          x.report("forward (T=#{seq_len})") do
            Torch.no_grad do
              model.call(input_ids)
            end
          end
        end

        x.compare!
      end

      puts
    end

    def benchmark_generation(model, tokenizer)
      puts '📊 Generation Benchmarks (KEY METRICS)'
      puts '-' * 80

      prompt = 'Once upon a time'
      prompt_ids = tokenizer.encode(prompt)

      # Throughput benchmark (like Python nanochat)
      puts "\nGeneration throughput:"

      max_tokens = 100

      # Without KV cache
      puts "\n  Without KV cache:"
      start_time = Time.now
      tokens_generated = 0

      Torch.no_grad do
        model.generate(prompt_ids, max_tokens: max_tokens, temperature: 0.0, seed: 42) do |_token|
          tokens_generated += 1
        end
      end

      elapsed = Time.now - start_time
      throughput = tokens_generated / elapsed

      puts "    Generated #{tokens_generated} tokens in #{elapsed.round(2)}s"
      puts "    Throughput: #{throughput.round(2)} tokens/second"

      # With KV cache (Engine)
      puts "\n  With KV cache (Engine):"
      engine = Nanochat::Engine.new(model: model, tokenizer: tokenizer)

      start_time = Time.now
      output = engine.generate(prompt, max_tokens: max_tokens, temperature: 0.0, seed: 42)
      elapsed = Time.now - start_time

      output_tokens = tokenizer.encode(output)
      tokens_generated = output_tokens.length - prompt_ids.length
      throughput = tokens_generated / elapsed

      puts "    Generated #{tokens_generated} tokens in #{elapsed.round(2)}s"
      puts "    Throughput: #{throughput.round(2)} tokens/second"

      # Sampling strategy overhead
      puts "\nSampling strategy overhead:"

      Benchmark.ips do |x|
        x.config(time: 5, warmup: 2)

        x.report('greedy (temp=0.0)') do
          Torch.no_grad do
            model.generate(prompt_ids, max_tokens: 10, temperature: 0.0, seed: 42) { |_| }
          end
        end

        x.report('temperature sampling (0.8)') do
          Torch.no_grad do
            model.generate(prompt_ids, max_tokens: 10, temperature: 0.8, seed: 42) { |_| }
          end
        end

        x.report('top-k sampling (k=40)') do
          Torch.no_grad do
            model.generate(prompt_ids, max_tokens: 10, temperature: 0.8, top_k: 40, seed: 42) { |_| }
          end
        end

        x.compare!
      end

      puts
    end
  end
end

# Parse command line options
options = {}
OptionParser.new do |opts|
  opts.banner = 'Usage: benchmark.rb [options]'

  opts.on('--model PATH', 'Path to model checkpoint') do |path|
    options[:model_path] = path
  end

  opts.on('--tokenizer PATH', 'Path to tokenizer directory') do |path|
    options[:tokenizer_path] = path
  end

  opts.on('-h', '--help', 'Show this help message') do
    puts opts
    exit
  end
end.parse!

# Run benchmarks
runner = NanochatBenchmark::Runner.new(**options)
runner.run_all

puts '=' * 80
puts 'Benchmark complete!'
puts
puts 'Note: For comparison with Python nanochat, focus on:'
puts '  - Generation throughput (tokens/second)'
puts '  - KV cache speedup factor'
puts '  - Forward pass latency'
