# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive hands-on tutorial for understanding adversarial attacks
- Interactive Jupyter notebook with step-by-step CNN implementation
- PyTorch CNN architecture built from scratch
- Fast Gradient Sign Method (FGSM) adversarial attack implementation
- Complete training pipeline with DataLoaders and optimization
- Forward hooks for monitoring CNN internals
- Detailed visualizations of adversarial perturbations
- 8,000-word companion article explaining security implications
- Real-world examples and defense strategies
- Performance analysis across different epsilon values
- CONTRIBUTING.md with participation guidelines

### Research Findings
- 98.45% accuracy on clean MNIST data
- 82.14% accuracy at epsilon 0.10 (16pp degradation with invisible perturbations)
- 41.23% accuracy at epsilon 0.20 (worse than random)
- Demonstrates critical gap between test accuracy and adversarial robustness

[Unreleased]: https://github.com/scthornton/understanding-adversarial-attacks-mnist/commits/main
