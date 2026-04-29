$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

pyinstaller .\packaging\pyinstaller.spec --noconfirm

Write-Host ""
Write-Host "Built EXE in: dist\\PlantDetect\\PlantDetect.exe"
Write-Host "Run and open: http://127.0.0.1:5000"

