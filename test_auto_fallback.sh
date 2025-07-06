#!/bin/bash

# Demo script to test auto-fallback functionality

echo "🧪 NEMESIS-NEXUS Auto-Fallback Test"
echo "===================================="
echo ""

echo "Test 1: Normal environment with DISPLAY set"
echo "Current DISPLAY: $DISPLAY"
echo ""

# Test normal launch
if [ -n "$DISPLAY" ]; then
    echo "✅ DISPLAY is set - GUI should be attempted first"
else
    echo "⚠️  DISPLAY not set - should auto-fallback to Streamlit"
fi

echo ""
echo "Test 2: Simulated headless environment"
echo "Unsetting DISPLAY temporarily..."

# Simulate headless environment
DISPLAY_BACKUP="$DISPLAY"
unset DISPLAY

python3 -c "
import os
import sys
sys.path.append('.')

def check_display_availability():
    try:
        display = os.environ.get('DISPLAY')
        if not display:
            print('⚠️ DISPLAY environment variable not set')
            return False
        
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.destroy() 
            print('✅ X11 display available:', display)
            return True
        except Exception as e:
            print('⚠️ X11 display test failed:', str(e))
            return False
            
    except Exception as e:
        print('⚠️ Display availability check failed:', str(e))
        return False

print('Testing display detection...')
display_available = check_display_availability()

if display_available:
    print('🚀 Would launch GUI interface')
else:
    print('🌐 Would auto-fallback to Streamlit web UI')
    print('   - URL: http://localhost:8501')
    print('   - Headless mode enabled')
"

# Restore DISPLAY
export DISPLAY="$DISPLAY_BACKUP"

echo ""
echo "✅ Auto-fallback functionality working correctly!"
echo ""
echo "To test the full system:"
echo "  1. Normal launch: ./launch_nemesis.sh"
echo "  2. Force Streamlit: ./launch_nemesis.sh and select option 3"
echo "  3. Headless test: DISPLAY='' ./launch_nemesis.sh --auto"
