# frozen_string_literal: true

require 'test_helper'
require 'tempfile'

class CheckpointManagerTest < Minitest::Test
  def test_load_raises_for_missing_file
    error = assert_raises(ArgumentError) do
      Nanochat::CheckpointManager.load('/nonexistent/path.pt')
    end
    assert_match(/not found/i, error.message)
  end

  def test_load_raises_for_directory
    Dir.mktmpdir do |dir|
      # Trying to load a directory instead of a file raises an IO error
      assert_raises(Errno::EISDIR) do
        Nanochat::CheckpointManager.load(dir)
      end
    end
  end

  def test_save_creates_checkpoint_file
    Dir.mktmpdir do |dir|
      checkpoint_path = File.join(dir, 'test_checkpoint.pt')
      state_dict = {'layer1.weight' => Torch.randn([3, 3])}

      Nanochat::CheckpointManager.save(checkpoint_path, state_dict: state_dict)

      assert_path_exists checkpoint_path
    end
  end

  def test_save_and_load_round_trip
    Dir.mktmpdir do |dir|
      checkpoint_path = File.join(dir, 'test_checkpoint.pt')
      original_tensor = Torch.randn([3, 3])
      state_dict = {'layer1.weight' => original_tensor}

      Nanochat::CheckpointManager.save(checkpoint_path, state_dict: state_dict)
      loaded_state = Nanochat::CheckpointManager.load(checkpoint_path)

      assert_kind_of Hash, loaded_state
      assert loaded_state.key?('model')
      assert_equal original_tensor.shape, loaded_state['model']['layer1.weight'].shape
    end
  end

  def test_save_with_nested_state_dict
    Dir.mktmpdir do |dir|
      checkpoint_path = File.join(dir, 'nested_checkpoint.pt')
      state_dict = {
        'encoder.weight' => Torch.randn([10, 10]),
        'encoder.bias' => Torch.randn([10]),
        'decoder.weight' => Torch.randn([10, 10])
      }

      Nanochat::CheckpointManager.save(checkpoint_path, state_dict: state_dict)
      loaded_state = Nanochat::CheckpointManager.load(checkpoint_path)

      assert_equal 3, loaded_state['model'].keys.size
      assert loaded_state['model'].key?('encoder.weight')
      assert loaded_state['model'].key?('encoder.bias')
      assert loaded_state['model'].key?('decoder.weight')
    end
  end

  def test_save_creates_parent_directories
    Dir.mktmpdir do |dir|
      nested_path = File.join(dir, 'subdir', 'nested', 'checkpoint.pt')
      state_dict = {'weight' => Torch.randn([2, 2])}

      Nanochat::CheckpointManager.save(nested_path, state_dict: state_dict)

      assert_path_exists nested_path
    end
  end
end
