# Setup Environment for The Kairos Engine (Gemma 4 GGUF + XGBoost)

Write-Host "⚡ Setting up Python Virtual Environment for The Kairos Engine..." -ForegroundColor Cyan

# Create virtual environment if not existing
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "[OK] Created venv." -ForegroundColor Green
}

# Activate
.\venv\Scripts\Activate.ps1

# Upgrade Pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python CPU wheel
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Train XGBoost risk model
python build_crash_predictor.py

Write-Host "`n🚀 Environment setup complete! Run 'python main.py' to launch Kairos Engine." -ForegroundColor Green
