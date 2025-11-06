# Understanding Adversarial Attacks Through MNIST

**Hands-On Tutorial: Building CNNs and Testing Adversarial Robustness with PyTorch**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository demonstrates the fragility of deep learning models through hands-on experimentation with adversarial attacks. You'll build a CNN from scratch, train it to 98% accuracy on MNIST, then watch it collapse to 41% accuracy when faced with imperceptible adversarial perturbations.

**What makes neural networks so vulnerable?** Decision boundaries learned during training are often extremely close to legitimate examples. A tiny push in the right direction—changing pixels by amounts invisible to humans—can flip predictions completely. This isn't a theoretical curiosity. It's a critical security issue affecting autonomous vehicles, medical imaging systems, facial recognition, and fraud detection.

## What Makes This Tutorial Unique

This tutorial specifically focuses on **Convolutional Neural Networks (CNNs)** for adversarial robustness—not just applying FGSM to pre-trained models:

- **Build CNNs from scratch** - Learn conv layers, pooling, and architecture design, not just use them
- **Understand spatial hierarchies** - See how feature maps at each layer respond to adversarial perturbations
- **Monitor CNN internals** - Use PyTorch forward hooks to inspect activations throughout the network
- **Complete training pipeline** - Master DataLoaders, optimization, and proper evaluation
- **Educational focus** - 8,000-word companion article explaining real-world security implications
- **Visual intuition** - Detailed visualizations showing how convolutional features break down under attack

While FGSM is a general attack method, this tutorial teaches you how CNNs specifically learn spatial patterns and fail under adversarial conditions. This makes it ideal for computer vision practitioners who need to understand both architecture design and security testing.

## What You'll Learn

### PyTorch Fundamentals
- Build CNNs using `torch.nn.Module`
- Implement proper training loops with backpropagation
- Use DataLoaders for efficient batching and preprocessing
- Monitor model internals with forward hooks
- Save and load model checkpoints

### Security & Robustness
- Generate adversarial examples using Fast Gradient Sign Method (FGSM)
- Measure the gap between clean accuracy and adversarial robustness
- Visualize imperceptible perturbations that fool neural networks
- Understand why 98% test accuracy doesn't mean production-ready
- Learn defense strategies and their limitations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/scthornton/understanding-adversarial-attacks-mnist.git
cd understanding-adversarial-attacks-mnist

# Install dependencies
pip install torch torchvision matplotlib numpy jupyter
```

### Run the Tutorial

```bash
# Launch Jupyter notebook
jupyter notebook adversarial-mnist-tutorial.ipynb
```

The notebook is self-contained and walks through every step with detailed explanations.

## Key Results

When we apply FGSM attacks with different epsilon values:

| Epsilon | Perturbation Visibility | Accuracy | Interpretation |
|---------|-------------------------|----------|----------------|
| 0.00 | No attack (clean) | 98.45% | Baseline performance |
| 0.05 | Barely detectable | 95.32% | Small degradation |
| 0.10 | Still invisible to humans | 82.14% | Significant vulnerability |
| 0.20 | Slightly noticeable | 41.23% | Worse than random guessing |
| 0.30 | Visible noise | 18.67% | Model is broken |

**Critical Insight:** At epsilon 0.1 (roughly 3% of the normalized pixel range), perturbations remain invisible to humans but reduce accuracy by 16 percentage points. The model that appeared production-ready based on test accuracy collapses under adversarial conditions.

## Repository Structure

```
understanding-adversarial-attacks-mnist/
├── README.md                              # This file
├── adversarial-mnist-tutorial.ipynb       # Main educational notebook
├── adversarial-mnist-article-edited.md    # Deep-dive article on adversarial attacks
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── images/                                # Visualizations and results
    ├── architecture-diagram.png
    ├── adversarial-examples.png
    └── robustness-curve.png
```

## Notebook Contents

### Part 1: Setup and Architecture
- Import libraries and set up environment
- Build a simple CNN from scratch
- Understand the architecture flow and parameter counts

### Part 2: Data Loading
- Load and preprocess MNIST dataset
- Create efficient DataLoaders with batching and shuffling
- Visualize training samples

### Part 3: Training Pipeline
- Implement training loop with proper gradient handling
- Track loss and accuracy over epochs
- Evaluate on test set to measure generalization

### Part 4: Monitoring with Hooks
- Use PyTorch forward hooks to inspect activations
- Monitor internal representations during inference
- Debug and understand what the network learns

### Part 5: Adversarial Attacks
- Implement Fast Gradient Sign Method (FGSM)
- Test robustness across different epsilon values
- Visualize clean images, perturbations, and adversarial examples

### Part 6: Security Analysis
- Understand the gap between clean and adversarial accuracy
- Learn defense strategies (adversarial training, certified defenses)
- Best practices for deploying models in adversarial environments

## Companion Article

The repository includes `adversarial-mnist-article-edited.md`, a comprehensive 8,000-word article that covers:
- Why adversarial vulnerabilities matter beyond MNIST
- Real-world incidents (autonomous vehicles, facial recognition, medical imaging)
- Current defense strategies and their limitations
- Practical guidance for ML engineers deploying models in production
- The path forward for secure machine learning

## Real-World Implications

### Autonomous Vehicles
Adversarial patches on stop signs can make object detection models fail to recognize them. Researchers demonstrated this using physical stickers arranged in computed patterns—the vehicle's camera captures the scene, and the neural network misclassifies critical road signs.

### Medical Imaging
A 2019 study showed adversarial perturbations that make malignant tumors invisible to cancer detection networks while making benign tissue appear malignant. The perturbations are subtle enough that radiologists don't notice them.

### Facial Recognition
Specially designed eyeglass frames or adversarial makeup patterns can bypass security checkpoints, evade recognition, or impersonate other individuals. Every access control system relying on facial recognition carries this vulnerability.

### Financial Fraud Detection
Transaction patterns that should trigger fraud alerts can be perturbed slightly—changing amounts, timing, or account relationships—to preserve fraudulent intent while evading detection.

## Defense Strategies

The notebook covers several defense approaches:

1. **Adversarial Training**: Include adversarial examples in training data (most effective but reduces clean accuracy)
2. **Defensive Distillation**: Smooth decision boundaries with softmax temperature (bypassed by strong attacks)
3. **Input Preprocessing**: Apply transformations to remove perturbations (defeated by adaptive attackers)
4. **Certified Defenses**: Provide provable robustness guarantees (severe accuracy/performance tradeoffs)
5. **Detection Methods**: Identify adversarial inputs before classification (evaded by adaptive attacks)

**Key Takeaway**: No defense provides perfect protection. Production systems require defense in depth—layered security combining multiple independent mechanisms.

## Prerequisites

**Required Knowledge:**
- Basic Python programming
- Fundamental understanding of neural networks
- Familiarity with NumPy

**Recommended (but not required):**
- Prior PyTorch experience
- Calculus (for understanding gradients)
- Linear algebra (for understanding matrix operations)

**Time Commitment:**
- Complete tutorial: 2-3 hours
- Reading companion article: 30 minutes
- Exercises and experimentation: 1-2 hours

## Exercises for Further Learning

### Beginner
1. Add learning rate scheduling with `torch.optim.lr_scheduler.StepLR`
2. Plot training curves (loss and accuracy over epochs)
3. Experiment with different batch sizes and observe the impact

### Intermediate
4. Add batch normalization after convolutional layers
5. Compare SGD vs Adam vs AdamW optimizers
6. Implement PGD attack (multi-step iterative FGSM)

### Advanced
7. Implement adversarial training (train on mix of clean and adversarial examples)
8. Export model to ONNX format for deployment
9. Create a robustness curve plotting accuracy vs epsilon for multiple attack methods

## Contributing

Contributions are welcome! This is an educational project focused on teaching adversarial machine learning through practical examples.

**Ways to contribute:**
- Report bugs or unclear explanations
- Suggest additional exercises or examples
- Add new attack implementations (PGD, C&W, adversarial patches)
- Improve visualizations
- Add defense implementations

Please open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tutorial in your research or teaching, please cite:

```bibtex
@misc{thornton2025adversarial,
  author = {Thornton, Scott},
  title = {Understanding Adversarial Attacks Through MNIST},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/scthornton/understanding-adversarial-attacks-mnist}
}
```

## Resources

### Papers
- [Explaining and Harnessing Adversarial Examples (FGSM)](https://arxiv.org/abs/1412.6572) - Goodfellow et al., 2014
- [Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)](https://arxiv.org/abs/1706.06083) - Madry et al., 2017
- [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) - Cohen et al., 2019

### Tools & Benchmarks
- [RobustBench](https://robustbench.github.io/) - Adversarial robustness benchmark
- [CleverHans](https://github.com/cleverhans-lab/cleverhans) - Adversarial attack library
- [Foolbox](https://github.com/bethgelab/foolbox) - Python toolbox for adversarial attacks

### Documentation
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [PyTorch API Reference](https://pytorch.org/docs/stable/index.html)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## Acknowledgments

- **MNIST Dataset**: LeCun, Cortes, and Burges
- **FGSM Attack**: Ian Goodfellow and colleagues (2014)
- **PyTorch Framework**: Facebook AI Research
- Inspired by the need to understand adversarial vulnerabilities before deploying ML systems in production

## Contact

**Scott Thornton**
- Website: [perfecxion.ai](https://perfecxion.ai)
- GitHub: [@scthornton](https://github.com/scthornton)
- LinkedIn: [Scott Thornton](https://linkedin.com/in/scthornton)

---

**⚠️ Security Disclaimer**: This tutorial is for educational purposes. Understanding adversarial attacks is critical for building secure ML systems. Always test robustness before deploying models in security-critical applications.

**⭐ If you found this tutorial helpful, please consider giving it a star!**
