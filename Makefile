# BTC Quantitative Analysis - Makefile
# Quick commands for both environments

.PHONY: help setup-venv run-analysis docker-build docker-run test clean

help:
	@echo "BTC Quantitative Analysis - Available Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  setup-venv     - Create and setup virtual environment"
	@echo "  activate-venv   - Activate virtual environment (manual)"
	@echo ""
	@echo "Development (venv):"
	@echo "  run-analysis    - Run enhanced analysis"
	@echo "  run-test        - Run quick test analysis"
	@echo "  jupyter         - Start Jupyter Lab"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run analysis in Docker"
	@echo "  docker-jupyter  - Start Jupyter in Docker"
	@echo ""
	@echo "Testing:"
	@echo "  test            - Run all tests"
	@echo "  test-features   - Test enhanced features"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           - Clean generated files"
	@echo "  help            - Show this help"

# Environment Setup
setup-venv:
	@echo "Setting up virtual environment..."
	python -m venv btc-quant-env
	@echo "Virtual environment created. Activate with:"
	@echo "  source btc-quant-env/bin/activate  # Linux/Mac"
	@echo "  btc-quant-env\\Scripts\\activate     # Windows"
	@echo "Then run: pip install -r requirements.txt"

activate-venv:
	@echo "Please activate the virtual environment manually:"
	@echo "  source btc-quant-env/bin/activate  # Linux/Mac"
	@echo "  btc-quant-env\\Scripts\\activate     # Windows"

# Development Commands (venv)
run-analysis:
	@echo "Running enhanced analysis..."
	python enhanced_main.py

run-test:
	@echo "Running quick test analysis..."
	python enhanced_main.py --test

jupyter:
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Docker Commands
docker-build:
	@echo "Building Docker image..."
	docker build -t btc-quant-alpha .

docker-run:
	@echo "Running analysis in Docker..."
	docker-compose up btc-quant

docker-jupyter:
	@echo "Starting Jupyter in Docker..."
	docker-compose up jupyter

docker-full:
	@echo "Running full Docker stack..."
	docker-compose up

# Testing
test:
	@echo "Running all tests..."
	pytest tests/ -v

test-features:
	@echo "Testing enhanced features..."
	python test_enhanced_features.py

# Utilities
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .pytest_cache/
	rm -rf results/*.png
	rm -rf exports/*.csv
	rm -rf exports/visualizations/*.html
	@echo "Clean complete!"

# Quick start for new users
quick-start:
	@echo "BTC Quantitative Analysis - Quick Start"
	@echo "====================================="
	@echo ""
	@echo "Option 1: Virtual Environment (Recommended for Development)"
	@echo "  make setup-venv"
	@echo "  source btc-quant-env/bin/activate  # Linux/Mac"
	@echo "  pip install -r requirements.txt"
	@echo "  make run-analysis"
	@echo ""
	@echo "Option 2: Docker (Production Demo)"
	@echo "  make docker-build"
	@echo "  make docker-run"
	@echo ""
	@echo "For Jupyter development:"
	@echo "  make jupyter  # with venv"
	@echo "  make docker-jupyter  # with Docker" 