param(
    [string]$HostName = "213.173.107.85",
    [int]$Port = 19355,
    [string]$User = "root",
    [string]$KeyPath = "~/.ssh/id_ed25519",
    [string]$RemoteWorkspace = "/workspace",
    [string]$BackupTag = "incremental",
    [switch]$CopyLogs,
    [switch]$Commit,
    [switch]$Push
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

function Convert-RemoteToLocalPath {
    param(
        [string]$BasePath,
        [string]$RemoteRelativePath
    )
    $normalized = $RemoteRelativePath.Trim().TrimStart("./").Replace("/", "\")
    return Join-Path $BasePath $normalized
}

$resolvedKey = Resolve-SshKeyPath -InputPath $KeyPath
$repoRoot = (Get-Location).Path
$backupRoot = Join-Path $repoRoot "artifacts\runpod_backups"
$destRoot = Join-Path $backupRoot $BackupTag
$destModels = Join-Path $destRoot "models"
$destLogs = Join-Path $destRoot "logs"

New-Item -ItemType Directory -Force -Path $destModels | Out-Null
New-Item -ItemType Directory -Force -Path $destLogs | Out-Null

Write-Host "RunPod incremental backup"
Write-Host "  Host: $User@$HostName`:$Port"
Write-Host "  Backup tag: $BackupTag"
Write-Host "  Destination: $destRoot"
Write-Host ""

$findCmd = "cd $RemoteWorkspace && find models -type f -name metadata.json -printf '%h`n' | sort"
$remoteModelDirsRaw = & ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $resolvedKey -p "$Port" "$User@$HostName" $findCmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list remote model directories."
}

$remoteModelDirs = @($remoteModelDirsRaw | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$copied = New-Object System.Collections.Generic.List[string]
$skipped = New-Object System.Collections.Generic.List[string]

foreach ($remoteDir in $remoteModelDirs) {
    $localDir = Convert-RemoteToLocalPath -BasePath $destRoot -RemoteRelativePath $remoteDir
    if (Test-Path $localDir) {
        $skipped.Add($remoteDir)
        continue
    }

    $parent = Split-Path -Parent $localDir
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $remoteAbs = "$User@$HostName`:$RemoteWorkspace/$remoteDir"
    & scp -i $resolvedKey -P "$Port" -r $remoteAbs $parent
    if ($LASTEXITCODE -ne 0) {
        throw "Failed copying remote directory: $remoteDir"
    }
    $copied.Add($remoteDir)
}

if ($CopyLogs) {
    $logsToCopy = @("training.log", "training_sol.log", "training_btc_5m.log", "training_sol_5m.log")
    foreach ($logName in $logsToCopy) {
        $remoteLog = "$User@$HostName`:$RemoteWorkspace/logs/$logName"
        & scp -i $resolvedKey -P "$Port" $remoteLog $destLogs 2>$null
        $null = $LASTEXITCODE
    }
}

$readmePath = Join-Path $destRoot "README.md"
$sourceLine = "- source: {0}@{1}:{2}" -f $User, $HostName, $Port
$workspaceLine = "- workspace: {0}" -f $RemoteWorkspace
$readme = @(
    "# RunPod Backup - $BackupTag",
    "",
    $sourceLine,
    $workspaceLine,
    "- generated_utc: $([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    "- new_runs_copied: $($copied.Count)",
    "- existing_runs_skipped: $($skipped.Count)",
    "",
    "This folder is maintained by scripts/runpod_backup_incremental.ps1.",
    "Each run directory includes its own metadata.json for reproducibility."
)
$readme | Set-Content -Path $readmePath -Encoding UTF8

$manifestPath = Join-Path $destRoot "manifest_sha256.txt"
$manifestLines = Get-ChildItem -Recurse -File $destRoot | Where-Object { $_.Name -ne "manifest_sha256.txt" } | ForEach-Object {
    $hash = Get-FileHash -Algorithm SHA256 $_.FullName
    $rel = $_.FullName.Substring($destRoot.Length + 1).Replace("\", "/")
    "$($hash.Hash)  $rel"
}
$manifestLines | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "Backup result:"
Write-Host "  Copied runs: $($copied.Count)"
Write-Host "  Skipped runs: $($skipped.Count)"
if ($copied.Count -gt 0) {
    $copied | ForEach-Object { Write-Host "    + $_" }
}
Write-Host ""

if ($Commit -or $Push) {
    & git add $destRoot
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed."
    }

    $hasStaged = & git diff --cached --name-only
    if ($LASTEXITCODE -ne 0) {
        throw "git diff --cached failed."
    }

    if ($hasStaged) {
        $msg = "backup: runpod incremental sync ($BackupTag) copied=$($copied.Count)"
        & git commit -m $msg
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed."
        }
        if ($Push) {
            & git push origin main
            if ($LASTEXITCODE -ne 0) {
                throw "git push failed."
            }
        }
    } else {
        Write-Host "No new backup files staged; skipping commit/push."
    }
}
