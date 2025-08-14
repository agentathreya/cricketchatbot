#!/usr/bin/env python3
"""
Setup script for Ollama (Free LLM)
==================================

This script helps you set up Ollama for the cricket chatbot.
Ollama provides completely free local LLM inference.
"""

import subprocess
import sys
import requests
import time

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed!")
            return True
        else:
            print("❌ Ollama is not installed.")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed.")
        return False

def install_ollama():
    """Install Ollama based on the operating system."""
    import platform
    system = platform.system().lower()
    
    print(f"🔧 Installing Ollama for {system}...")
    
    if system == "darwin":  # macOS
        print("📥 Downloading Ollama for macOS...")
        subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'], shell=True)
    elif system == "linux":
        print("📥 Downloading Ollama for Linux...")
        subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'], shell=True)
    elif system == "windows":
        print("📥 Please download Ollama from: https://ollama.ai/download")
        print("   Then run this script again.")
        return False
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False
    
    return True

def start_ollama():
    """Start the Ollama service."""
    print("🚀 Starting Ollama service...")
    try:
        # Start Ollama in the background
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for Ollama to start
        print("⏳ Waiting for Ollama to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama is running!")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"   {i+1}/30 seconds...")
        
        print("❌ Ollama failed to start within 30 seconds.")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def pull_model(model_name="llama2"):
    """Pull a model from Ollama."""
    print(f"📥 Pulling {model_name} model...")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {model_name} model downloaded successfully!")
            return True
        else:
            print(f"❌ Failed to download {model_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def list_available_models():
    """List available models."""
    print("\n📋 Available models:")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model in models:
                print(f"   - {model['name']}")
        else:
            print("   No models found.")
    except:
        print("   Could not fetch models.")

def main():
    """Main setup function."""
    print("🏏 Cricket Chatbot - Ollama Setup")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n📥 Installing Ollama...")
        if not install_ollama():
            print("\n❌ Installation failed. Please install manually from https://ollama.ai/")
            return
    
    # Start Ollama
    if not start_ollama():
        print("\n❌ Failed to start Ollama. Please start manually with: ollama serve")
        return
    
    # Pull a model
    model_choice = input("\n🤖 Which model would you like to use? (llama2/mistral/codellama): ").strip()
    if not model_choice:
        model_choice = "llama2"
    
    if not pull_model(model_choice):
        print(f"\n❌ Failed to download {model_choice}. Trying llama2...")
        if not pull_model("llama2"):
            print("❌ Failed to download any model.")
            return
    
    # List available models
    list_available_models()
    
    print("\n🎉 Setup complete!")
    print("🚀 You can now run: streamlit run app.py")
    print(f"🤖 The chatbot will use the {model_choice} model locally (completely free!)")

if __name__ == "__main__":
    main()
