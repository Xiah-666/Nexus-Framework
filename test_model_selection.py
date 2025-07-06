#!/usr/bin/env python3
"""
Test script for model selection functionality
Tests the new model management endpoints and UI integration
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_model_endpoints():
    """Test the model management API endpoints"""
    
    print("🧪 Testing Model Management Endpoints")
    print("=" * 50)
    
    # Test 1: Get available models
    print("\n1. Testing /models endpoint...")
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models = response.json()["models"]
            print(f"✅ Found {len(models)} available models:")
            for model in models[:3]:  # Show first 3
                print(f"   - {model['name']} ({model.get('size', 0) / (1024**3):.1f} GB)")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting models: {e}")
    
    # Test 2: Get current model
    print("\n2. Testing /models/current endpoint...")
    try:
        response = requests.get(f"{API_URL}/models/current")
        if response.status_code == 200:
            current_model = response.json()["current_model"]
            print(f"✅ Current model: {current_model}")
        else:
            print(f"❌ Failed to get current model: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting current model: {e}")
    
    # Test 3: Test model selection (if models available)
    print("\n3. Testing /models/select endpoint...")
    try:
        # First get available models
        models_response = requests.get(f"{API_URL}/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            if models:
                test_model = models[0]["name"]
                print(f"   Attempting to select: {test_model}")
                
                response = requests.post(f"{API_URL}/models/select", 
                                       json={"model_name": test_model})
                if response.status_code == 200:
                    print(f"✅ Successfully selected model: {test_model}")
                else:
                    print(f"❌ Failed to select model: {response.status_code}")
            else:
                print("⚠️  No models available for testing selection")
        else:
            print("⚠️  Cannot test selection - no models endpoint")
    except Exception as e:
        print(f"❌ Error testing model selection: {e}")
    
    # Test 4: Test LLM chat with model
    print("\n4. Testing /llm/chat with model selection...")
    try:
        # Get current model first
        current_response = requests.get(f"{API_URL}/models/current")
        if current_response.status_code == 200:
            current_model = current_response.json()["current_model"]
            
            chat_payload = {
                "prompt": "Say hello and tell me what model you are.",
                "model": current_model
            }
            
            response = requests.post(f"{API_URL}/llm/chat", json=chat_payload)
            if response.status_code == 200:
                chat_response = response.json()["response"]
                print(f"✅ LLM Response: {chat_response[:100]}...")
            else:
                print(f"❌ Failed LLM chat: {response.status_code}")
        else:
            print("⚠️  Cannot test chat - no current model")
    except Exception as e:
        print(f"❌ Error testing LLM chat: {e}")

def test_ollama_connection():
    """Test direct Ollama connection"""
    
    print("\n🔗 Testing Direct Ollama Connection")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models")
            for model in models[:3]:
                print(f"   - {model.get('name', 'Unknown')}")
        else:
            print(f"❌ Ollama returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama (not running or wrong port)")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")

def main():
    """Main test function"""
    print("🚀 Nemesis-Nexus Model Selection Test Suite")
    print("=" * 60)
    
    # Test Ollama connection first
    test_ollama_connection()
    
    # Test API endpoints
    test_model_endpoints()
    
    print("\n🏁 Test Complete!")
    print("\n💡 Tips:")
    print("   - Start the API with: python nemesis_web_api.py")
    print("   - Start the UI with: streamlit run nemesis_web_ui.py")
    print("   - Make sure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()
