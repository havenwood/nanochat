# frozen_string_literal: true

require 'fileutils'

module Nanochat
  # Common utilities
  module Common
    class << self
      def device
        @device ||= if Torch::CUDA.available?
                      Torch.device('cuda')
                    elsif defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
                      Torch.device('mps')
                    else
                      Torch.device('cpu')
                    end
      end

      def seed(seed_value)
        Torch.manual_seed(seed_value)
        Torch::CUDA.manual_seed_all(seed_value) if Torch::CUDA.available?
      end

      def default_cache_dir = ENV.fetch('NANOCHAT_BASE_DIR') { File.expand_path('~/.cache/nanochat') }

      def ensure_dir(path)
        FileUtils.mkdir_p(path) unless File.directory?(path)
        path
      end
    end
  end
end
