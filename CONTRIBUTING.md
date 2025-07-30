# Contributing to BTC Quant

Thank you for your interest in contributing to BTC Quant! This document outlines the standards and guidelines for contributing to this project.

## 🎯 **Project Philosophy**

BTC Quant demonstrates **professional software engineering practices** in quantitative trading. We prioritize:

- **Code Quality**: Clean, tested, documented code
- **Honest Communication**: Realistic performance claims
- **Professional Standards**: Industry best practices
- **Educational Value**: Learning opportunities for contributors

## 🚀 **Quick Start for Contributors**

### **Prerequisites**
- Python 3.11+
- Git
- Docker (optional, for containerized development)

### **Setup Development Environment**

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/BTC-Quant.git
   cd BTC-Quant
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Verify Setup**
   ```bash
   python -m pytest tests/ -v
   python -m flake8 src/ tests/
   ```

## 📋 **Development Standards**

### **Code Style**
- **Line Length**: 88 characters (Black default)
- **Formatting**: Use Black for code formatting
- **Imports**: Use isort for import organization
- **Linting**: Flake8 with zero errors

### **Testing Requirements**
- **Coverage**: Maintain 80%+ test coverage
- **New Features**: Must include corresponding tests
- **Bug Fixes**: Must include regression tests
- **Integration**: Test complete pipeline functionality

### **Documentation**
- **Docstrings**: All functions must have clear docstrings
- **Type Hints**: Use type hints for all function parameters
- **README**: Keep documentation up to date
- **Comments**: Explain complex logic, not obvious code

## 🔧 **Development Workflow**

### **1. Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

### **2. Make Changes**
- Write clean, tested code
- Follow the established patterns
- Add appropriate tests
- Update documentation

### **3. Quality Checks**
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
flake8 src/ tests/

# Run tests
python -m pytest tests/ -v

# Check coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### **4. Commit Changes**
```bash
git add .
git commit -m "feat: Add your feature description"
```

### **5. Push and Create PR**
```bash
git push origin feature/your-feature-name
```

## 🧪 **Testing Guidelines**

### **Test Structure**
```python
def test_function_name():
    """Test description of what is being tested."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_value
```

### **Test Categories**
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test complete pipelines
- **Edge Cases**: Test boundary conditions
- **Error Handling**: Test exception scenarios

### **Test Data**
- Use static, reproducible test data
- Mock external dependencies (API calls, file I/O)
- Avoid random data in tests

## 📊 **Performance Standards**

### **Code Performance**
- **Time Complexity**: Document O(n) complexity
- **Memory Usage**: Consider memory efficiency
- **Scalability**: Test with larger datasets

### **Trading Performance**
- **Realistic Claims**: Don't overstate results
- **Transaction Costs**: Account for fees when possible
- **Risk Metrics**: Include drawdown and volatility
- **Backtesting**: Use walk-forward validation

## 🎨 **Visualization Standards**

### **Chart Requirements**
- **Professional Style**: Use consistent color schemes
- **Clear Labels**: Descriptive titles and axis labels
- **Accessibility**: Consider colorblind-friendly palettes
- **Export Quality**: High DPI for publications

### **Dashboard Guidelines**
- **Layout**: Logical organization of charts
- **Interactivity**: Consider user interaction needs
- **Responsive**: Adapt to different screen sizes
- **Performance**: Optimize for large datasets

## 🔍 **Code Review Process**

### **Review Checklist**
- [ ] Code follows style guidelines
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Security implications are addressed
- [ ] Error handling is appropriate

### **Review Comments**
- **Constructive**: Focus on improvement, not criticism
- **Specific**: Point to exact lines and suggest fixes
- **Educational**: Explain why changes are needed
- **Respectful**: Maintain professional tone

## 🚨 **Bug Reports**

### **Bug Report Template**
```markdown
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python: [e.g., 3.11.4]
- Dependencies: [relevant versions]

**Additional Context**
Any other relevant information
```

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
**Feature Description**
Clear description of the requested feature

**Use Case**
Why this feature is needed

**Proposed Implementation**
How you suggest implementing it

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 📈 **Performance Improvements**

### **Acceptable Improvements**
- **Algorithm Optimization**: Better time/space complexity
- **Memory Management**: Reduced memory usage
- **Parallel Processing**: Multi-threading where appropriate
- **Caching**: Intelligent caching strategies
- **Data Structures**: More efficient data structures

### **Unacceptable Changes**
- **Over-Engineering**: Unnecessary complexity
- **Premature Optimization**: Optimizing before profiling
- **Breaking Changes**: Without proper migration path
- **Performance Claims**: Without proper validation

## 🔒 **Security Guidelines**

### **Data Security**
- **API Keys**: Never commit API keys or secrets
- **User Data**: Don't collect unnecessary personal data
- **Encryption**: Use secure protocols for data transmission
- **Validation**: Validate all inputs and outputs

### **Code Security**
- **Dependencies**: Keep dependencies updated
- **Vulnerabilities**: Address security warnings
- **Input Validation**: Sanitize all inputs
- **Error Messages**: Don't expose sensitive information

## 📚 **Documentation Standards**

### **Code Documentation**
```python
def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of return values
        risk_free_rate: Annual risk-free rate (default: 0.02)
    
    Returns:
        float: Sharpe ratio
    
    Raises:
        ValueError: If returns list is empty
    
    Example:
        >>> returns = [0.01, -0.02, 0.03, -0.01]
        >>> calculate_sharpe_ratio(returns)
        0.1234
    """
```

### **README Updates**
- **New Features**: Document new functionality
- **API Changes**: Update usage examples
- **Dependencies**: Update requirements
- **Configuration**: Document new settings

## 🎯 **Contribution Types**

### **Code Contributions**
- **Bug Fixes**: Fix issues in existing code
- **Feature Additions**: Add new functionality
- **Performance Improvements**: Optimize existing code
- **Refactoring**: Improve code structure

### **Documentation Contributions**
- **README Updates**: Improve project documentation
- **Code Comments**: Add clarifying comments
- **API Documentation**: Document function interfaces
- **Tutorials**: Create learning resources

### **Testing Contributions**
- **Test Coverage**: Add missing tests
- **Edge Cases**: Test boundary conditions
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical functions

## 🏆 **Recognition**

### **Contributor Recognition**
- **Commit History**: All contributors are acknowledged
- **Release Notes**: Contributors listed in releases
- **Documentation**: Contributors mentioned in README
- **Community**: Active contributors invited to discussions

### **Quality Standards**
- **Professional Code**: Production-ready quality
- **Comprehensive Testing**: Thorough test coverage
- **Clear Documentation**: Well-documented changes
- **Community Focus**: Benefits the broader community

## 📞 **Getting Help**

### **Communication Channels**
- **Issues**: Use GitHub issues for bugs and features
- **Discussions**: Use GitHub discussions for questions
- **Code Review**: Ask questions in pull requests
- **Documentation**: Check existing documentation first

### **Before Asking**
- [ ] Check existing issues and discussions
- [ ] Read the relevant documentation
- [ ] Try to reproduce the issue
- [ ] Provide minimal reproduction steps

## 📄 **License**

By contributing to BTC Quant, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to BTC Quant!** 🚀

Your contributions help make this project a better learning resource and demonstration of professional software engineering practices. 