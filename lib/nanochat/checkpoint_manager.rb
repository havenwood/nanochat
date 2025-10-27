# frozen_string_literal: true

module Nanochat
  # Checkpoint loading and saving
  module CheckpointManager
    class << self
      def load(path)
        raise ArgumentError, "Checkpoint not found: #{path}" unless File.exist?(path)

        Torch.load(path)
      end

      def save(path, model: nil, state_dict: nil, optimizer: nil, config: nil, **metadata)
        raise ArgumentError, 'Must provide either model: or state_dict:' if model.nil? && state_dict.nil?

        FileUtils.mkdir_p(File.dirname(path))

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
