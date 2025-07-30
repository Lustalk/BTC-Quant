# Contributing to BTC Trading Strategy

Thank you for your interest in contributing to this enterprise-grade quantitative trading project! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/btc-trading-strategy.git`
3. **Setup development environment**:
   ```bash
   make setup-venv
   source btc-quant-env/bin/activate  # Linux/Mac
   make install-dev
   ```
4. **Create a feature branch**: `git checkout -b feature/your-feature-name`
5. **Make your changes**
6. **Run quality checks**: `make format lint type-check test`
7. **Commit your changes**: `git commit -m "feat: Add your feature description"`
8. **Push to your fork**: `git push origin feature/your-feature-name`
9. **Create a Pull Request**

## 📋 Development Guidelines

### Code Quality Standards

- **Type Hints**: All functions must have type annotations
- **Documentation**: All public functions must have docstrings
- **Testing**: New features must include unit tests
- **Formatting**: Code must be formatted with Black
- **Linting**: Code must pass Flake8 checks
- **Type Checking**: Code must pass MyPy checks

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat: Add Monte Carlo simulation module
fix: Correct off-by-one error in data windowing
docs: Update README with new CLI options
test: Add integration tests for feature engineering
```

### Testing Requirements

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows
- **Coverage**: Maintain at least 80% test coverage
- **Performance Tests**: Test for performance regressions

Run tests with:
```bash
make test              # Run all tests
make test-cov          # Run with coverage report
pytest tests/ -m unit  # Run only unit tests
pytest tests/ -m slow  # Run slow tests
```

### Code Quality Checks

Before submitting a PR, ensure your code passes all quality checks:

```bash
make format      # Format code with Black
make lint        # Lint with Flake8
make type-check  # Type check with MyPy
make test        # Run all tests
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🏗️ Project Structure

```
src/
├── data_pipeline.py          # Data acquisition & preprocessing
├── feature_engineering.py    # Technical indicators
├── modeling.py              # ML models
├── validation.py            # Walk-forward validation
├── evaluation.py            # Performance metrics
├── risk_management.py       # Risk controls
├── monte_carlo_simulation.py # Risk analysis
└── professional_visualizations.py # Dashboards

tests/
├── test_data_pipeline.py    # Data pipeline tests
├── test_feature_engineering.py # Feature engineering tests
├── test_modeling.py         # Model tests
├── test_validation.py       # Validation tests
└── test_integration.py      # Integration tests
```

## 🔧 Development Workflow

### 1. Environment Setup

```bash
# Create virtual environment
make setup-venv

# Activate environment
source btc-quant-env/bin/activate  # Linux/Mac
# btc-quant-env\Scripts\activate   # Windows

# Install dependencies
make install-dev
```

### 2. Development

```bash
# Run quick test
make run-test

# Start Jupyter for exploration
make jupyter

# Run full analysis
make run-analysis
```

### 3. Quality Assurance

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run tests
make test

# Run all quality checks
make pre-commit
```

### 4. Docker Development

```bash
# Build and run with Docker
docker-compose up --build

# Run Jupyter in Docker
docker-compose up jupyter
```

## 📊 Performance Considerations

When contributing to this quantitative trading project:

- **Data Efficiency**: Use vectorized operations with NumPy/Pandas
- **Memory Management**: Avoid loading large datasets into memory
- **Caching**: Cache expensive computations
- **Parallelization**: Use multiprocessing for independent operations
- **Profiling**: Profile code for performance bottlenecks

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, dependency versions
2. **Reproduction**: Steps to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Logs**: Relevant error messages and logs
5. **Minimal Example**: Minimal code to reproduce the issue

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: What alternatives have you considered?
4. **Implementation**: Any thoughts on implementation approach?

## 📝 Documentation

When contributing documentation:

- Use clear, concise language
- Include code examples
- Update both docstrings and README
- Add type hints to all examples
- Include error handling in examples

## 🤝 Code Review Process

1. **Automated Checks**: All PRs must pass CI/CD checks
2. **Code Review**: At least one maintainer must approve
3. **Testing**: New features must include tests
4. **Documentation**: New features must include documentation
5. **Performance**: Changes must not significantly impact performance

## 📈 Performance Benchmarks

When adding new features, consider:

- **Execution Time**: How long does it take to run?
- **Memory Usage**: How much memory does it use?
- **Scalability**: How does it perform with larger datasets?
- **Accuracy**: Does it improve model performance?

## 🚨 Security Considerations

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Validate all inputs
- Handle exceptions gracefully
- Log security-relevant events

## 📞 Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the README and docstrings first

Thank you for contributing to making this project better! 🎉 