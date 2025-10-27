# frozen_string_literal: true

require_relative 'lib/nanochat/version'

Gem::Specification.new do |spec|
  spec.name = 'nanochat'
  spec.version = Nanochat::VERSION
  spec.authors = ['Shannon Skipper']
  spec.email = ['shannonskipper@gmail.com']

  spec.summary = 'Ruby port of nanochat - minimal ChatGPT implementation'
  spec.description = 'A minimal, hackable LLM implementation with inference capabilities. Port of Karpathy\'s nanochat.'
  spec.homepage = 'https://github.com/havenwood/nanochat'
  spec.required_ruby_version = '>= 3.4.0'
  spec.license = 'MIT'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = "#{spec.homepage}/tree/main"
  spec.metadata['changelog_uri'] = "#{spec.homepage}/releases"
  spec.metadata['bug_tracker_uri'] = "#{spec.homepage}/issues"
  spec.metadata['documentation_uri'] = "https://rubydoc.info/gems/#{spec.name}/#{spec.version}"
  spec.metadata['rubygems_mfa_required'] = 'true'

  spec.files = Dir['lib/**/*.rb', 'lib/**/*.html', 'lib/**/*.svg', 'bin/*', 'LICENSE', 'README.md']
  spec.executables = ['nanochat-setup'] # Only end-user facing executable
  spec.require_paths = ['lib']

  spec.add_dependency 'tokenizers', '~> 0.6.1'
  spec.add_dependency 'torch-rb', '~> 0.22.1'
end
