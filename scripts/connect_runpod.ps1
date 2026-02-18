param(
    [string]$HostName = "103.196.86.56",
    [int]$Port = 10801,
    [string]$User = "root",
    [string]$KeyPath = "~/.ssh/id_ed25519",
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SshKeyPath {
    param([string]$InputPath)

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        throw "Key path is empty."
    }

    $candidate = $InputPath.Trim()
    $homeNormalized = $HOME.TrimEnd("\", "/")

    if ($candidate -match '^~[\\/]') {
        $candidate = Join-Path $homeNormalized $candidate.Substring(2)
    } elseif ($candidate -eq "~") {
        $candidate = $homeNormalized
    }

    if ($candidate.StartsWith("/")) {
        $candidate = Join-Path $homeNormalized $candidate.TrimStart("/")
    }

    $candidate = $candidate -replace "/", "\"

    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path (Get-Location).Path $candidate
    }

    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "SSH key not found at: $candidate"
    }

    return (Resolve-Path -LiteralPath $candidate).Path
}

$resolvedKey = Resolve-SshKeyPath -InputPath $KeyPath

$sshArgs = @(
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-i", $resolvedKey,
    "-p", "$Port",
    "$User@$HostName"
)

if ($CheckOnly) {
    $sshArgs = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15"
    ) + $sshArgs
    $sshArgs += @("echo CONNECTED && whoami && hostname && nvidia-smi -L")
}

Write-Host "Using key: $resolvedKey"
Write-Host "Target: $User@$HostName`:$Port"
Write-Host ""

& ssh @sshArgs
exit $LASTEXITCODE
