#!/usr/bin/env ruby
# frozen_string_literal: true

# Web chat server for nanochat
# Mirrors python-nanochat/scripts/chat_web.py functionality
#
# Usage:
#   ruby examples/chat_web.rb [options]
#
# Options:
#   -c, --checkpoint PATH   Path to model checkpoint (default: ~/.cache/nanochat/model.pt)
#   -t, --tokenizer PATH    Path to tokenizer directory (default: ~/.cache/nanochat/tokenizer)
#   -T, --temperature TEMP  Default temperature (default: 0.8)
#   -k, --top-k K          Default top-k sampling (default: 50)
#   -m, --max-tokens N     Default max tokens (default: 512)
#   -p, --port PORT        Port to run server on (default: 8000)
#   -H, --host HOST        Host to bind to (default: 0.0.0.0)
#   -h, --help             Show this help message

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'nanochat'
require 'roda'
require 'json'
require 'logger'
require 'optparse'

# Configuration
Config = Struct.new(:checkpoint, :tokenizer_dir, :temperature, :top_k, :max_tokens, :port, :host, keyword_init: true)

config = Config.new(
  checkpoint: File.expand_path('~/.cache/nanochat/model.pt'),
  tokenizer_dir: File.expand_path('~/.cache/nanochat/tokenizer'),
  temperature: 0.8,
  top_k: 50,
  max_tokens: 512,
  port: 8000,
  host: '0.0.0.0'
)

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32_000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = OptionParser.new do |opts|
  opts.banner = 'Usage: chat_web.rb [OPTIONS]'
  opts.separator ''
  opts.separator 'Web chat server for nanochat'
  opts.separator ''
  opts.separator 'Options:'

  opts.on('-c', '--checkpoint PATH', 'Path to model checkpoint') do |path|
    config.checkpoint = File.expand_path(path)
  end

  opts.on('-t', '--tokenizer PATH', 'Path to tokenizer directory') do |path|
    config.tokenizer_dir = File.expand_path(path)
  end

  opts.on('-T', '--temperature TEMP', Float, "Default temperature (default: #{config.temperature})") do |temp|
    config.temperature = temp
  end

  opts.on('-k', '--top-k K', Integer, "Default top-k (default: #{config.top_k})") do |k|
    config.top_k = k
  end

  opts.on('-m', '--max-tokens N', Integer, "Default max tokens (default: #{config.max_tokens})") do |n|
    config.max_tokens = n
  end

  opts.on('-p', '--port PORT', Integer, "Port (default: #{config.port})") do |port|
    config.port = port
  end

  opts.on('-H', '--host HOST', "Host (default: #{config.host})") do |host|
    config.host = host
  end

  opts.on('-h', '--help', 'Show this help message') do
    puts opts
    exit
  end
end

parser.parse!

# Initialize logger
logger = Logger.new($stdout)
logger.level = Logger::INFO
logger.formatter = proc do |_severity, datetime, _progname, msg|
  "#{datetime.strftime('%Y-%m-%d %H:%M:%S')} - #{msg}\n"
end

# Validate files exist
unless File.exist?(config.checkpoint)
  puts "❌ Checkpoint not found: #{config.checkpoint}"
  exit 1
end

tokenizer_file = File.join(config.tokenizer_dir, 'tokenizer.json')
unless File.exist?(tokenizer_file)
  puts "❌ Tokenizer not found: #{tokenizer_file}"
  exit 1
end

# Load model
puts '🔄 Loading model and tokenizer...'
model_config = Nanochat::Config.from_checkpoint(
  Nanochat::CheckpointManager.load(config.checkpoint)
)
model = Nanochat::GPT.from_checkpoint(config.checkpoint, model_config)
tokenizer = Nanochat::Tokenizer.from_directory(config.tokenizer_dir)
device = Nanochat::Common.device
Nanochat::Engine.new(model:, tokenizer:, device:)

puts "✅ Model loaded (device: #{device})"

# Special tokens
BOS = tokenizer.bos_token_id
USER_START = tokenizer.token_to_id('<|user_start|>')
USER_END = tokenizer.token_to_id('<|user_end|>')
ASSISTANT_START = tokenizer.token_to_id('<|assistant_start|>')
ASSISTANT_END = tokenizer.token_to_id('<|assistant_end|>')

# Roda web application
class NanoChatWeb < Roda
  plugin :json
  plugin :streaming
  plugin :halt
  plugin :head
  plugin :slash_path_empty
  plugin :public, root: File.expand_path('assets', __dir__)

  route do |r|
    # Serve static files from assets/ (logo.svg, etc)
    r.public

    # GET / - Serve chat UI
    r.root do
      ui_path = File.expand_path('assets/ui.html', __dir__)
      html = File.read(ui_path)
      # Replace API URL to use same origin
      html.gsub!(
        'const API_URL = `http://${window.location.hostname}:8000`;',
        "const API_URL = '';"
      )
      response['Content-Type'] = 'text/html'
      html
    end

    # GET /health - Health check
    r.get 'health' do
      {
        status: 'healthy',
        device: device.to_s,
        model_config: {
          vocab_size: model_config.vocab_size,
          n_embd: model_config.n_embd,
          n_layer: model_config.n_layer
        }
      }
    end

    # GET /stats - Statistics
    r.get 'stats' do
      {
        device: device.to_s,
        config: {
          temperature: config.temperature,
          top_k: config.top_k,
          max_tokens: config.max_tokens
        }
      }
    end

    # POST /chat/completions - Chat API
    r.post 'chat/completions' do
      request_body = JSON.parse(r.body.read, symbolize_names: true)

      # Validate request
      messages = request_body[:messages] || []
      temperature = request_body[:temperature] || config.temperature
      top_k = request_body[:top_k] || config.top_k
      max_tokens = request_body[:max_tokens] || config.max_tokens

      # Apply abuse prevention limits
      r.halt(400, {error: 'Too many messages'}.to_json) if messages.length > MAX_MESSAGES_PER_REQUEST

      total_length = messages.sum { |msg| msg[:content].to_s.length }
      r.halt(400, {error: 'Conversation too long'}.to_json) if total_length > MAX_TOTAL_CONVERSATION_LENGTH

      messages.each do |msg|
        r.halt(400, {error: 'Message too long'}.to_json) if msg[:content].to_s.length > MAX_MESSAGE_LENGTH
      end

      temperature = temperature.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE)
      top_k = top_k.clamp(MIN_TOP_K, MAX_TOP_K)
      max_tokens = max_tokens.clamp(MIN_MAX_TOKENS, MAX_MAX_TOKENS)

      # Log conversation
      logger.info('=' * 20)
      messages.each do |msg|
        logger.info("[#{msg[:role].upcase}]: #{msg[:content]}")
      end
      logger.info('-' * 20)

      # Build conversation tokens
      conversation_tokens = [BOS]
      messages.each do |msg|
        case msg[:role]
        when 'user'
          conversation_tokens << USER_START
          conversation_tokens.concat(tokenizer.encode(msg[:content]))
          conversation_tokens << USER_END
        when 'assistant'
          conversation_tokens << ASSISTANT_START
          conversation_tokens.concat(tokenizer.encode(msg[:content]))
          conversation_tokens << ASSISTANT_END
        end
      end
      conversation_tokens << ASSISTANT_START

      # Stream response
      response['Content-Type'] = 'text/event-stream'
      response['Cache-Control'] = 'no-cache'
      response['Connection'] = 'keep-alive'

      stream do |out|
        accumulated_tokens = []
        last_clean_text = ''

        begin
          engine.generate_stream(
            conversation_tokens,
            max_tokens:,
            temperature:,
            top_k:
          ) do |_token_text, token_id|
            # Stop on assistant_end or bos
            break if [ASSISTANT_END, BOS].include?(token_id)

            accumulated_tokens << token_id
            current_text = tokenizer.decode(accumulated_tokens)

            # Only emit if doesn't end with replacement character (incomplete UTF-8)
            unless current_text.end_with?('�')
              new_text = current_text[last_clean_text.length..]
              unless new_text.empty?
                data = {token: new_text, device: device.to_s}.to_json
                out << "data: #{data}\n\n"
                last_clean_text = current_text
              end
            end
          end

          # Log assistant response
          logger.info("[ASSISTANT] (#{device}): #{last_clean_text}")
          logger.info('=' * 20)

          out << "data: #{({done: true}.to_json)}\n\n"
        rescue StandardError => e
          logger.error("Error: #{e.message}")
          logger.error(e.backtrace.join("\n"))
          out << "data: #{({error: e.message}.to_json)}\n\n"
        end
      end
    end
  end
end

# Start server with Falcon
puts "\n🚀 NanoChat Web Server"
puts '=' * 50
puts "Server running at http://#{config.host}:#{config.port}"
puts 'Open in browser to start chatting!'
puts '=' * 50
puts

require 'falcon'
require 'async'

Async do
  Falcon::Server.run(
    NanoChatWeb.freeze.app,
    endpoint: Falcon::Endpoint.parse("http://#{config.host}:#{config.port}")
  )
end.wait
