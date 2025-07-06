#!/bin/bash

# NEMESIS-NEXUS Launcher Script
# Advanced Multi-Agent AI Cybersecurity Framework

echo "🚀 NEMESIS-NEXUS - Advanced Multi-Agent AI Cybersecurity Framework"
echo "=================================================================="

# Function to check if X/display is available
check_display() {
    if [ -z "$DISPLAY" ]; then
        echo "⚠️  X/desktop display not available (DISPLAY not set)"
        return 1
    fi
    
    # Try to test X11 connection
    if command -v xset &> /dev/null; then
        if ! xset q &> /dev/null; then
            echo "⚠️  X/desktop display not available (xset test failed)"
            return 1
        fi
    fi
    
    echo "✅ X/desktop display available: $DISPLAY"
    return 0
}

# Function to start Streamlit UI with proper URLs
start_streamlit() {
    echo "🌐 Starting Streamlit web UI..."
    echo ""
    
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
    
    uv run streamlit run nemesis_web_ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
}

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

echo "✅ Dependencies checked"
echo ""

# Auto-detect: Always use Streamlit as default (regardless of X availability)
if [ "$1" = "--auto" ] || [ "$1" = "" ]; then
    echo "🚀 Auto-launching NEMESIS-NEXUS with Streamlit web UI (default)..."
    start_streamlit
    exit 0
fi

# Display available modes
echo "Available modes:"
echo "  0) Auto Mode - Streamlit web UI (default)"
echo "  1) CLI Interface - Command line interface"
echo "  2) Web API - FastAPI web interface" 
echo "  3) Web UI - Streamlit web interface"
echo "  4) GUI Mode - Desktop GUI interface"
echo "  5) Test LLM - Test AI model connectivity"
echo "  6) Show Status - Display system status"
echo ""

read -p "Select mode (0-6, or Enter for auto): " mode

# Default to auto mode if no selection
if [ -z "$mode" ]; then
    mode=0
fi

case $mode in
    0)
        echo "🚀 Auto Mode - Launching Streamlit web UI (default)..."
        start_streamlit
        ;;
    1)
        echo "🖥️  Starting CLI Interface..."
        uv run python cli_interface.py
        ;;
    2)
        echo "🌐 Starting Web API on http://localhost:8000..."
        uv run python pray.py --web
        ;;
    3)
        echo "🌐 Starting Web UI on http://localhost:8501..."
        start_streamlit
        ;;
    4)
        echo "🖥️  Starting GUI Mode..."
        if check_display; then
            echo "🚀 Launching NEMESIS-NEXUS with GUI interface..."
            uv run python pray.py
        else
            echo "⚠️  GUI mode requires X display. Falling back to Streamlit..."
            start_streamlit
        fi
        ;;
    5)
        echo "🤖 Testing LLM connectivity..."
        uv run python test_llm_simple.py
        ;;
    6)
        echo "📊 System Status:"
        echo "=================="
        echo "Ollama Models:"
        uv run python -c "import ollama; client = ollama.Client(); models = client.list()['models']; [print(f'  - {m[\"name\"]}') for m in models]" 2>/dev/null || echo "  Error: Could not connect to Ollama"
        echo ""
        echo "System Resources:"
        uv run python -c "import psutil; print(f'  CPU: {psutil.cpu_count()} cores'); print(f'  Memory: {psutil.virtual_memory().total // (1024**3)}GB')" 2>/dev/null || echo "  Error: Could not get system info"
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac
