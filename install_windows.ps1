# Windows install script for Nemesis-Nexus
python -m venv venv
./venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# TODO: Automate Ollama installation (manual for now)
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found; please install manually: https://ollama.com/download/windows"
}
if (!(Test-Path "config.json")) { Copy-Item "config.json.sample" "config.json" }
Write-Host '✅ Windows Installation complete. Activate environment with: .\venv\Scripts\Activate.ps1'
