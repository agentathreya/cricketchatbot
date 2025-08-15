#!/usr/bin/env python3
"""
Setup script for IPL Cricket Analytics Chatbot with Groq API
============================================================

This script helps set up the environment and dependencies.
"""

import os
import sys
import subprocess
import urllib.request

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_data_file():
    """Check if cricket data file exists"""
    if os.path.exists("cricket_data.csv"):
        print("✅ Cricket data file found")
        return True
    else:
        print("⚠️  cricket_data.csv not found")
        print("   Please ensure the data file is in the project directory")
        return False

def setup_env_file():
    """Help user set up environment file"""
    if os.path.exists(".env"):
        print("✅ .env file exists")
        return True
    
    print("⚠️  .env file not found")
    print("\n🔑 Setting up Groq API key:")
    print("1. Visit https://console.groq.com/ to get your FREE API key")
    print("2. Copy your API key")
    
    api_key = input("3. Paste your GROQ_API_KEY here (or press Enter to skip): ").strip()
    
    if api_key:
        with open(".env", "w") as f:
            f.write(f"GROQ_API_KEY={api_key}\n")
        print("✅ .env file created successfully!")
        return True
    else:
        print("⚠️  You can create .env file manually later")
        print("   Format: echo 'GROQ_API_KEY=your_key_here' > .env")
        return False

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")
    try:
        import streamlit
        import pandas 
        import langchain_groq
        from dotenv import load_dotenv
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False

def main():
    """Main setup function"""
    print("🏏 IPL Cricket Analytics Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Check data file
    check_data_file()
    
    # Setup environment
    setup_env_file()
    
    print("\n🎉 Setup completed!")
    print("\n🚀 To run the app:")
    print("   streamlit run app.py")
    print("\n💡 Don't forget to add your GROQ_API_KEY to .env file!")
    print("   Get it FREE at: https://console.groq.com/")

if __name__ == "__main__":
    main()
