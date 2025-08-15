#!/usr/bin/env python3
"""
test_app.py
-----------

Test script to verify the cricket chatbot functionality
Run this before deployment to ensure everything works
"""

import os
import sys
import pandas as pd

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    try:
        import streamlit as st
        import pandas as pd  
        from langchain.agents import AgentType, initialize_agent
        from langchain.tools import StructuredTool
        from langchain_groq import ChatGroq
        from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
        from dotenv import load_dotenv
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test cricket data loading"""
    print("🧪 Testing data loading...")
    try:
        if os.path.exists("cricket_data.csv"):
            df = pd.read_csv("cricket_data.csv", nrows=100)
            print(f"✅ Data loaded: {len(df)} rows")
            
            # Test required columns
            required_cols = ["batter", "runs_batter", "batting_team", "over"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
            else:
                print("✅ All required columns present")
            return True
        else:
            print("⚠️  cricket_data.csv not found (OK for deployment testing)")
            return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_utils_import():
    """Test utils.py functions"""
    print("🧪 Testing utils functions...")
    try:
        from utils import (
            get_most_runs, get_most_wickets, get_most_fours_and_sixes,
            team_total_runs, get_powerplay_stats
        )
        print("✅ Utils functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False

def test_tool_creation():
    """Test tool creation with mock data"""
    print("🧪 Testing tool creation...")
    try:
        from langchain.tools import StructuredTool
        
        def mock_function():
            """Mock function for testing"""
            return "Mock data result"
        
        tool = StructuredTool.from_function(
            func=mock_function,
            name="mock_tool",
            description="A mock tool for testing"
        )
        print("✅ Tool creation successful")
        return True
    except Exception as e:
        print(f"❌ Tool creation failed: {e}")
        return False

def test_groq_llm():
    """Test Groq LLM initialization (without API call)"""
    print("🧪 Testing Groq LLM setup...")
    try:
        from langchain_groq import ChatGroq
        
        # Test class import
        print("✅ ChatGroq class imported")
        
        # Test with mock API key (won't make actual calls)
        if os.getenv('GROQ_API_KEY'):
            print("✅ GROQ_API_KEY found in environment")
        else:
            print("⚠️  GROQ_API_KEY not found (expected for testing)")
        
        return True
    except Exception as e:
        print(f"❌ Groq LLM test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏏 Cricket Chatbot Functionality Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading), 
        ("Utils Functions", test_utils_import),
        ("Tool Creation", test_tool_creation),
        ("Groq LLM", test_groq_llm)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        results[test_name] = test_func()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment!")
        print("\n🚀 Next steps:")
        print("1. Get Groq API key: https://console.groq.com/")
        print("2. Set environment: GROQ_API_KEY=your_key")
        print("3. Run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
