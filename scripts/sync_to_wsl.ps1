param(
    [string]$Source = (Get-Location).Path,
    [string]$Target = "~/projects/sarych-lm"
)

$ErrorActionPreference = "Stop"

if ($Target.StartsWith("~/")) {
    $WslUser = (wsl whoami).Trim()
    $Target = "/home/$WslUser/" + $Target.Substring(2)
}

function Convert-ToWslPath([string]$WindowsPath) {
    $resolved = (Resolve-Path $WindowsPath).Path
    $drive = $resolved.Substring(0, 1).ToLower()
    $rest = $resolved.Substring(2).Replace("\", "/")
    return "/mnt/$drive$rest"
}

$sourceWsl = Convert-ToWslPath $Source
$excludeArgs = @(
    "--exclude", ".venv/",
    "--exclude", "runs/",
    "--exclude", "__pycache__/",
    "--exclude", ".pytest_cache/",
    "--exclude", ".mypy_cache/",
    "--exclude", ".ruff_cache/",
    "--exclude", ".git/",
    "--exclude", "*.pt",
    "--exclude", "*.bin",
    "--exclude", "*.npy",
    "--exclude", "*.npz"
)

Write-Host "Source: $Source"
Write-Host "WSL source: $sourceWsl"
Write-Host "WSL target: $Target"

wsl bash -lc "mkdir -p '$Target'"
$hasRsync = wsl bash -lc "command -v rsync >/dev/null 2>&1; echo `$?"
if ($hasRsync.Trim() -eq "0") {
    $excludeText = ($excludeArgs | ForEach-Object { "'" + $_ + "'" }) -join " "
    wsl bash -lc "rsync -av $excludeText '$sourceWsl/' '$Target/'"
} else {
    Write-Host "rsync is not installed in WSL; using tar stream fallback without deleting target files."
    wsl bash -lc "cd '$sourceWsl' && tar --exclude='.venv' --exclude='runs' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.git' --exclude='*.pt' --exclude='*.bin' --exclude='*.npy' --exclude='*.npz' -cf - . | (cd '$Target' && tar -xf -)"
}

Write-Host ""
Write-Host "Next commands inside WSL:"
Write-Host "cd ~/projects/sarych-lm"
Write-Host "source .venv/bin/activate"
Write-Host "python scripts/env_report.py --output runs/v0_1_synthetic_sanity/env_report.txt"
Write-Host "pytest -q"
Write-Host "python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml"
