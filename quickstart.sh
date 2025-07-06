#!/bin/bash

# NEMESIS-NEXUS Quick Start Script
# Launches Streamlit web interface directly

echo "🚀 NEMESIS-NEXUS Quick Start"
echo "============================"
echo "🌐 Launching Streamlit Web UI..."
echo ""

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install UV first."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Get local IP for network access
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")

echo "📱 Access the web interface at:"
echo "   🏠 Local:   http://localhost:8501"
echo "   🌍 Network: http://$LOCAL_IP:8501"
echo ""
echo "⚠️  AUTHORIZED SECURITY TESTING ONLY ⚠️"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Launch Streamlit
streamlit run nemesis_web_ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
