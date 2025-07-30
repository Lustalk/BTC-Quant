"""
Setup script for Professional Bitcoin Trading Analysis System.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def setup_environment():
    """Set up environment variables."""
    print("🔧 Setting up environment...")
    
    # Create .env file from example if it doesn't exist
    if not os.path.exists('.env') and os.path.exists('env.example'):
        try:
            with open('env.example', 'r') as f:
                env_content = f.read()
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("✅ Created .env file from env.example")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    return True

def start_infrastructure():
    """Start Docker infrastructure."""
    print("🐳 Starting infrastructure...")
    
    # Check if Docker is available
    try:
        subprocess.run("docker --version", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Docker not found. Please install Docker first.")
        return False
    
    # Start infrastructure
    return run_command("docker-compose up -d postgres redis", "Starting infrastructure")

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    return run_command("python test_setup.py", "Running tests")

def main():
    """Main setup function."""
    print("🚀 Professional Bitcoin Trading Analysis - Setup")
    print("=" * 50)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Environment Setup", setup_environment),
        ("Dependency Installation", install_dependencies),
        ("Infrastructure Start", start_infrastructure),
        ("Test Suite", run_tests)
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        print(f"\n📋 Step {successful_steps + 1}/{total_steps}: {step_name}")
        if step_func():
            successful_steps += 1
        else:
            print(f"❌ Setup failed at step: {step_name}")
            break
    
    print("\n" + "=" * 50)
    print(f"📊 Setup Results: {successful_steps}/{total_steps} steps completed")
    
    if successful_steps == total_steps:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Start the application: python main.py")
        print("2. Visit the dashboard: http://localhost:8000")
        print("3. View API documentation: http://localhost:8000/api/docs")
        print("4. Check monitoring: http://localhost:9090 (Prometheus)")
        print("5. View analytics: http://localhost:3000 (Grafana)")
    else:
        print("⚠️  Setup incomplete. Please check the errors above and try again.")
    
    return successful_steps == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 