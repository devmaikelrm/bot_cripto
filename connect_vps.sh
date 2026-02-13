#!/bin/bash
export VPS_HOST="100.71.91.32"
export VPS_USER="maikelrm95"

# Do NOT store the password in this script or in git.
if [[ -z "${VPS_PASS:-}" ]]; then
  read -r -s -p "Password for ${VPS_USER}@${VPS_HOST}: " VPS_PASS
  echo ""
  export VPS_PASS
fi

python scripts/vps_check.py --fix

# Optional: interactive mode
# python vps.py
