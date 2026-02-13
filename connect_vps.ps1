$env:VPS_HOST="100.71.91.32"
$env:VPS_USER="maikelrm95"
# Do NOT store the password in this script or in git.
# We'll ask for it at runtime if it's not already set in the environment.
if (-not $env:VPS_PASS -or $env:VPS_PASS.Trim().Length -eq 0) {
  $secure = Read-Host "Password for $($env:VPS_USER)@$($env:VPS_HOST)" -AsSecureString
  $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try {
    $env:VPS_PASS = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) | Out-Null
  }
}

# Health-check + auto-fix (systemd timers/services, links, logs, journal)
python scripts\vps_check.py --fix

# Optional: interactive shell (uncomment if you want to drop into VPS command mode)
# python vps.py
