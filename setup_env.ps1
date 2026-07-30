# Setup Environment for Khumbu Engine (Gemma-4 on RTX 4050)

$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment 'venv'..."
python -m venv venv

Write-Host "Activating virtual environment..."
$env:VIRTUAL_ENV = "$PWD\venv"
$env:Path = "$PWD\venv\Scripts;$env:Path"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
# We install torch first manually to ensure cu121 is picked up correctly if requirements.txt gives issues,
# but our requirements.txt has the extra-index-url. Let's just use pip install -r requirements.txt.
pip install -r requirements.txt

Write-Host "Environment setup complete!"
Write-Host "To activate in the future, run: .\venv\Scripts\Activate.ps1"
