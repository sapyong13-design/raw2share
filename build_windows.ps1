if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}
Write-Host "Activating virtual environment and installing requirements..."
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
Write-Host "Running tests..."
$env:PYTHONPATH="src"
pytest
if ($LASTEXITCODE -ne 0) {
    Write-Error "Tests failed! Aborting build."
    exit $LASTEXITCODE
}
Write-Host "Building RAW2Share executable..."
pyinstaller --noconfirm --onedir --windowed --name="RAW2Share" --paths="src" src/main.py
Write-Host "Build complete! Output is in dist/RAW2Share/RAW2Share.exe"
