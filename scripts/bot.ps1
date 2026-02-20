param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$venv = ".venv312"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Missing $python. Run scripts/bootstrap_local_windows.ps1 first."
}

& $python -m bot_cripto.cli @Args
exit $LASTEXITCODE
