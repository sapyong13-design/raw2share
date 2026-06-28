if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}
Write-Host "Activating virtual environment and installing requirements..."
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
Write-Host "Running RAW2Share..."
python src/main.py
