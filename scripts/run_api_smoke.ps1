param(
    [string]$Symbol = "BTC/USDT",
    [string]$Timeframe = "5m"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

powershell -ExecutionPolicy Bypass -File .\scripts\bot.ps1 api-smoke --symbol $Symbol --timeframe $Timeframe
