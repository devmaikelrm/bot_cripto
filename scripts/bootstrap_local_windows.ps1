param(
    [string]$PythonVersion = "3.12",
    [string]$VenvName = ".venv312"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Bootstrapping Windows environment..."
Write-Host "  Python: $PythonVersion"
Write-Host "  Venv:   $VenvName"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python 3.12+ first."
}

if (-not (Test-Path $VenvName)) {
    py -$PythonVersion -m venv $VenvName
}

$python = Join-Path $VenvName "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Venv python not found at $python"
}

& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -e ".[dev]"

Write-Host ""
Write-Host "Done."
Write-Host "Use this environment with:"
Write-Host "  .\$VenvName\Scripts\Activate.ps1"
Write-Host "Or run commands directly via:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\bot.ps1 info"
