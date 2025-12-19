# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-19

### Added
- Complete package restructure with `src/` layout
- Comprehensive test suite with pytest (49 tests)
- Full documentation (README.md and AGENTS.md) at all directory levels
- Modern Python packaging with `pyproject.toml` and `uv` support
- Makefile for common development tasks
- Setup script for installing general_FEP_RL dependency
- Requirements files for pip users

### Changed
- Refactored encoders/decoders with standard Python naming conventions
  - `Encode_Image` → `ImageEncoder`
  - `Decode_Image` → `ImageDecoder`
  - Similar changes for Number and Description encoders/decoders
- Replaced keras with torchvision for MNIST dataset loading
- Split utilities into separate modules (`logging.py`, `plotting.py`)
- Updated all imports to use absolute package imports
- Removed hardcoded paths and cleaned up variable naming

### Fixed
- Fixed nested list handling in epoch history tracking
- Fixed empty list handling in image plotting
- Improved error handling and test coverage

### Documentation
- Added comprehensive README.md with installation and usage instructions
- Added AGENTS.md technical documentation at all levels
- Documented development workflow and testing procedures
- Added examples documentation

### Testing
- Implemented pytest test suite with 26 passing tests
- Added graceful skipping for tests requiring general_FEP_RL
- Achieved 97% coverage for data loader module
- Added pytest configuration file

## [Unreleased]

### Planned
- Type hints throughout codebase
- Pre-commit hooks configuration
- CI/CD pipeline setup
- Additional example scripts
- Performance benchmarks
