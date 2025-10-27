# frozen_string_literal: true

require 'fileutils'

module Nanochat
  # Common utilities for device detection, seeding, etc.
  module Common
    class << self
      # Detect best available device (CUDA, MPS, or CPU)
      def device
        @device ||= if Torch::CUDA.available?
                      Torch.device('cuda')
                    elsif defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
                      Torch.device('mps')
                    else
                      Torch.device('cpu')
                    end
      end

      # Set random seed for reproducibility
      def seed(seed_value)
        Torch.manual_seed(seed_value)
        Torch::CUDA.manual_seed_all(seed_value) if Torch::CUDA.available?
      end

      # Default cache directory for models
      def default_cache_dir = ENV.fetch('NANOCHAT_BASE_DIR') { File.expand_path('~/.cache/nanochat') }

      # Ensure directory exists
      def ensure_dir(path)
        FileUtils.mkdir_p(path) unless File.directory?(path)
        path
      end
    end
  end
end
