# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path('../lib', __dir__)

ENV['MT_NO_PLUGINS'] = '1'

require 'nanochat'
require 'minitest/autorun'
