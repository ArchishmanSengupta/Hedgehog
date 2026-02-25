# Contributing to Hedgehog

We welcome contributions to Hedgehog! This document provides guidelines for contributing.

## Code of Conduct

Please be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create an issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

### Suggesting Features

1. Check existing issues and PRs
2. Create an issue with:
   - Clear description of the feature
   - Use cases
   - Proposed implementation (if any)

### Pull Requests

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. Make your changes:
   - Follow the existing code style
   - Add tests for new features
   - Update documentation

4. Commit with clear messages:
   ```bash
   git commit -m "Add amazing feature"
   ```

5. Push and create a PR:
   ```bash
   git push origin feature/amazing-feature
   ```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ArchishmanSengupta/hedgehog.git
cd hedgehog

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Code Style

- Use Black for formatting
- Use Ruff for linting
- Type hint where possible
- Docstrings for public APIs

## Testing

- Write tests for new features
- Ensure existing tests pass
- Test on multiple devices (CPU, CUDA, MPS)

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Include examples for new features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
