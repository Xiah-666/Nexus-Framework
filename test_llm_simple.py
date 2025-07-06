#!/usr/bin/env python3
"""
Simple test script to verify LLM functionality
"""
import asyncio
import ollama
from typing import Optional, Dict

class SimpleLLMInterface:
    def __init__(self):
        self.client = None
        self.history = []

    async def initialize(self):
        """Initialize Ollama client"""
        try:
            self.client = ollama.AsyncClient(host='http://localhost:11434')
            # Test connection
            await self.client.list()
            print("✅ Ollama client initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Ollama client: {e}")
            return False

    async def generate(self, prompt: str, model: str = "llama3:latest", temperature: float = 0.7) -> str:
        """Generate response using Ollama"""
        try:
            if not self.client:
                await self.initialize()
            
            response = await self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 500
                },
                stream=False
            )
            
            result = response['response']
            
            # Store in history
            entry = {
                "prompt": prompt,
                "response": result,
                "model": model
            }
            self.history.append(entry)
            
            return result
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"

    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception as e:
            print(f"❌ Failed to get models: {e}")
            return []

async def test_llm_functionality():
    """Test LLM functionality with available models"""
    print("🚀 Testing NEMESIS-NEXUS LLM Interface")
    print("=" * 50)
    
    llm = SimpleLLMInterface()
    
    # Initialize
    if not await llm.initialize():
        return False
    
    # Get available models
    models = llm.get_available_models()
    print(f"📋 Available models: {models}")
    
    if not models:
        print("❌ No models available")
        return False
    
    # Test with the first available model
    test_model = models[0]
    print(f"\n🤖 Testing with model: {test_model}")
    
    # Test basic functionality
    test_prompts = [
        "Test if LLM interface is working. Respond with: SYSTEM OPERATIONAL",
        "What is cybersecurity?",
        "Generate a simple Python script that prints 'Hello World'"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt[:50]}...")
        response = await llm.generate(prompt, model=test_model)
        print(f"🤖 Response: {response[:200]}...")
        
        if response and not response.startswith("Error:"):
            print("✅ Test passed")
        else:
            print("❌ Test failed")
    
    # Test specialized models if available
    specialized_models = [
        "huihui_ai/qwen3-abliterated:14b",
        "huihui_ai/jan-nano-abliterated:latest",
        "hf.co/mlabonne/gemma-3-12b-it-abliterated-v2-GGUF:Q8_0"
    ]
    
    for model in specialized_models:
        if model in models:
            print(f"\n🔓 Testing uncensored model: {model}")
            response = await llm.generate(
                "You are an AI assistant for cybersecurity research. Explain the importance of ethical hacking.",
                model=model
            )
            print(f"🤖 Response: {response[:200]}...")
            break
    
    print("\n🎯 LLM Testing Complete!")
    return True

if __name__ == "__main__":
    asyncio.run(test_llm_functionality())
