# frozen_string_literal: true

module Nanochat
  # Manages loading and saving model checkpoints
  module CheckpointManager
    class << self
      # Load a checkpoint from disk
      def load(path)
        raise ArgumentError, "Checkpoint not found: #{path}" unless File.exist?(path)

        # Load PyTorch checkpoint using torch-rb
        Torch.load(path)
      end

      # Save a checkpoint to disk
      # Can pass either a model object or a state_dict directly
      def save(path, model: nil, state_dict: nil, optimizer: nil, config: nil, **metadata)
        raise ArgumentError, 'Must provide either model: or state_dict:' if model.nil? && state_dict.nil?

        # Create parent directory if needed
        FileUtils.mkdir_p(File.dirname(path))

        # Convert symbol keys to strings for PyTorch compatibility
        model_dict = model ? model.state_dict : state_dict
        model_dict = convert_keys_to_strings(model_dict)

        data = {
          'model' => model_dict,
          'config' => config&.to_h&.transform_keys(&:to_s),
          **metadata.transform_keys(&:to_s)
        }
        data['optimizer'] = optimizer.state_dict if optimizer

        Torch.save(data, path)
      end

      private

      def convert_keys_to_strings(hash)
        hash.transform_keys(&:to_s).transform_values do |value|
          value.is_a?(Hash) ? convert_keys_to_strings(value) : value
        end
      end
    end
  end
end
