#!/usr/bin/env bash
set -e

# 1. Create virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Ensure Ollama is installed and service is running
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.ai/install.sh | sh
fi
sudo systemctl enable --now ollama

# 4. Create default config if missing
[ -f config.json ] || cp config.json.sample config.json

# 5. Make pray.py executable
chmod +x pray.py

echo "✅ Installation complete. Activate with: source venv/bin/activate"
echo "👉 Run: ./pray.py list-models"
