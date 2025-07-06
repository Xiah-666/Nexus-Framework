#!/bin/bash

# NEMESIS-NEXUS Web Interface Startup Script
# Launches Streamlit as the primary web interface
# Default web UI for NEMESIS-NEXUS

echo "🚀 Starting NEMESIS-NEXUS Web Interface..."
echo "════════════════════════════════════════════"
echo "🌐 Streamlit Web UI - Default Interface"
echo ""

# Check if ports are available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8000 is already in use (FastAPI backend may already be running)"
else
    echo "🔧 Starting FastAPI backend on port 8000..."
    uv run uvicorn nemesis_web_api:app --host 0.0.0.0 --port 8000 &
    FASTAPI_PID=$!
    echo "   FastAPI PID: $FASTAPI_PID"
fi

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is already in use (Streamlit may already be running)"
else
    # Wait a moment for FastAPI to start
    echo "⏳ Waiting for FastAPI to initialize..."
    sleep 3
    
    echo "🌐 Starting Streamlit frontend on port 8501..."
    uv run streamlit run nemesis_web_ui.py --server.port 8501 --server.address 0.0.0.0 &
    STREAMLIT_PID=$!
    echo "   Streamlit PID: $STREAMLIT_PID"
fi

echo ""
echo "✅ NEMESIS-NEXUS Web Interface is now running!"
echo ""
echo "🌐 Primary Access (Streamlit Web UI):"
echo "   🏠 Local:   http://localhost:8501"
echo "   🌍 Network: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "🔧 Additional Services:"
echo "   Backend API (FastAPI): http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "🔐 Default login credentials:"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "⚠️  AUTHORIZED SECURITY TESTING ONLY ⚠️"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping NEMESIS-NEXUS services..."
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null
        echo "   Stopped FastAPI backend"
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo "   Stopped Streamlit frontend"
    fi
    # Kill any remaining processes on the ports
    pkill -f "uvicorn nemesis_web_api:app" 2>/dev/null
    pkill -f "streamlit run nemesis_web_ui.py" 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
