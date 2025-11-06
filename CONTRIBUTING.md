# Contributing to Understanding Adversarial Attacks Through MNIST

Thank you for your interest in contributing! This project aims to make adversarial machine learning accessible through hands-on education.

## How to Contribute

### Reporting Issues

If you find bugs, unclear explanations, or have suggestions:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include context**: What were you trying to do? What happened? What did you expect?
4. **Add screenshots** or code snippets if relevant

### Suggesting Enhancements

We welcome suggestions for:
- Additional attack implementations (PGD, C&W, adversarial patches)
- New defense mechanisms
- Better visualizations
- Clearer explanations
- Additional exercises or examples
- Performance improvements

Open an issue with the "enhancement" label and describe:
- The problem you're trying to solve
- Your proposed solution
- Why this would benefit learners

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/understanding-adversarial-attacks-mnist.git
   cd understanding-adversarial-attacks-mnist
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Development Guidelines

**Code Style:**
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for all functions and classes
- Keep functions focused and concise

**Documentation:**
- Update README.md if adding new features
- Add inline comments for complex logic
- Include usage examples in docstrings

**Testing:**
- Test your changes with the full notebook
- Verify code runs on both CPU and GPU (if applicable)
- Check that visualizations render correctly

#### Example Contribution

```python
def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int
) -> torch.Tensor:
    """
    Generate adversarial examples using Projected Gradient Descent.

    Args:
        model: Neural network to attack
        images: Clean images (batch)
        labels: True labels
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for each iteration
        num_steps: Number of PGD iterations

    Returns:
        Adversarial examples

    Example:
        >>> adv_images = pgd_attack(model, images, labels,
        ...                         epsilon=0.3, alpha=0.01, num_steps=40)
    """
    # Implementation here
    pass
```

#### Commit Messages

Use clear, descriptive commit messages:

**Good:**
- `Add PGD attack implementation with documentation`
- `Fix epsilon scaling in FGSM visualization`
- `Update README with new defense strategies section`

**Bad:**
- `Update code`
- `Fix bug`
- `Changes`

#### Pull Request Process

1. **Update documentation** if you changed functionality
2. **Test thoroughly** on your local machine
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Open a Pull Request** from your fork to the main repository
5. **Describe your changes**:
   - What problem does this solve?
   - How did you test it?
   - Any breaking changes?
6. **Wait for review** - maintainers will provide feedback
7. **Make requested changes** if needed
8. **Celebrate!** 🎉 Once merged, you're a contributor!

### Areas Needing Help

**High Priority:**
- [ ] Add PGD (Projected Gradient Descent) attack
- [ ] Implement adversarial training example
- [ ] Create robustness curve visualization
- [ ] Add defense comparison section

**Medium Priority:**
- [ ] Export model to ONNX format example
- [ ] Add batch normalization variant of CNN
- [ ] Implement C&W (Carlini-Wagner) attack
- [ ] Create adversarial patch demo

**Nice to Have:**
- [ ] Add more visualizations
- [ ] Create video walkthrough
- [ ] Add unit tests
- [ ] Performance optimizations

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone, regardless of:
- Age, body size, disability, ethnicity, gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior includes:**
- Harassment, trolling, or derogatory comments
- Publishing others' private information
- Conduct that could reasonably be considered inappropriate

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Questions?

Feel free to open an issue with the "question" label or reach out directly:
- GitHub: [@scthornton](https://github.com/scthornton)
- Website: [perfecxion.ai](https://perfecxion.ai)

---

Thank you for contributing to making adversarial machine learning education accessible to everyone!
