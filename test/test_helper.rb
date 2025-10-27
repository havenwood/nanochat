# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)

# Disable minitest plugin autoloading (avoids loading incompatible gems from other Ruby versions)
ENV['MT_NO_PLUGINS'] = '1'

require 'nanochat'
require 'minitest/autorun'

# Test helpers can be added here if needed in the future
