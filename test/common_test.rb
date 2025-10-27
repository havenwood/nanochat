# frozen_string_literal: true

require 'test_helper'

class CommonTest < Minitest::Test
  def test_device_detection
    device = Nanochat::Common.device

    assert_kind_of Torch::Device, device
  end

  def test_device_is_valid_type
    device = Nanochat::Common.device
    device_str = device.to_s

    # Device string format is like 'device(type: "mps")'
    assert_match(/cpu|cuda|mps/, device_str)
  end

  def test_default_cache_dir_contains_nanochat
    cache_dir = Nanochat::Common.default_cache_dir

    assert_includes cache_dir, '.cache/nanochat'
  end

  def test_default_cache_dir_is_absolute_path
    cache_dir = Nanochat::Common.default_cache_dir

    assert_match(%r{^/}, cache_dir)
  end

  def test_default_cache_dir_respects_env_variable
    original_dir = ENV.fetch('NANOCHAT_BASE_DIR', nil)

    begin
      ENV['NANOCHAT_BASE_DIR'] = '/tmp/custom_nanochat'
      cache_dir = Nanochat::Common.default_cache_dir

      assert_equal '/tmp/custom_nanochat', cache_dir
    ensure
      if original_dir
        ENV['NANOCHAT_BASE_DIR'] = original_dir
      else
        ENV.delete('NANOCHAT_BASE_DIR')
      end
    end
  end

  def test_seed_does_not_raise
    assert_nil Nanochat::Common.seed(42)
  end

  def test_seed_with_zero
    assert_nil Nanochat::Common.seed(0)
  end

  def test_seed_with_large_number
    assert_nil Nanochat::Common.seed((2**31) - 1)
  end

  def test_seed_produces_deterministic_results
    Nanochat::Common.seed(42)
    tensor1 = Torch.randn([10])

    Nanochat::Common.seed(42)
    tensor2 = Torch.randn([10])

    assert_equal tensor1.to_a, tensor2.to_a
  end
end
