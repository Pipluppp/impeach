param(
    [switch]$Install,
    [switch]$EnrollGitHub,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$TaskName = "Impeach Senate Source Intake"
$Repository = "git@github.com:Pipluppp/impeach.git"
$AgentRoot = Join-Path $env:LOCALAPPDATA "impeach-source-agent"
$LogRoot = Join-Path $env:LOCALAPPDATA "impeach-source-logs"

if ($EnrollGitHub) {
    $token = Get-Clipboard -Raw
    try {
        if ($token -notmatch '^github_pat_') {
            throw "Copy the repository-scoped fine-grained GitHub token, then rerun with -EnrollGitHub."
        }
        $token.Trim() | gh auth login --hostname github.com --with-token
        if ($LASTEXITCODE -ne 0) { throw "GitHub CLI rejected the token." }
        gh auth status
    }
    finally {
        Set-Clipboard -Value ""
        $token = $null
    }
    exit 0
}

if ($Install) {
    $pwsh = (Get-Command pwsh.exe).Source
    $script = $MyInvocation.MyCommand.Path
    $action = New-ScheduledTaskAction `
        -Execute $pwsh `
        -Argument "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`"" `
        -WorkingDirectory (Split-Path $script)
    $trigger = New-ScheduledTaskTrigger -Daily -At "18:00"
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 15)
    $principal = New-ScheduledTaskPrincipal `
        -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive `
        -RunLevel Limited
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description "At 18:00 Manila time, preserve new official Senate journals locally and hand them to GitHub Actions." `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Get-ChildItem $LogRoot -Filter "*.log" -File |
    Where-Object LastWriteTime -lt (Get-Date).AddDays(-30) |
    Remove-Item -Force
$log = Join-Path $LogRoot ("source-intake-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

& {
    if (-not (Test-Path (Join-Path $AgentRoot ".git"))) {
        git clone $Repository $AgentRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not create the dedicated source-agent checkout." }
    }
    Push-Location $AgentRoot
    try {
        if (git status --porcelain) {
            throw "Dedicated source-agent checkout is dirty: $AgentRoot"
        }
        git fetch --prune origin
        git switch main
        git merge --ff-only origin/main
        if ($LASTEXITCODE -ne 0) { throw "Could not fast-forward the source agent to origin/main." }

        $python = Join-Path $AgentRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $python)) {
            py -3 -m venv (Join-Path $AgentRoot ".venv")
        }
        $requirementsHash = (Get-FileHash "pipeline/requirements.txt" -Algorithm SHA256).Hash
        $hashFile = Join-Path $AgentRoot ".venv\requirements.sha256"
        $installedHash = if (Test-Path $hashFile) { (Get-Content $hashFile -Raw).Trim() } else { "" }
        if ($installedHash -ne $requirementsHash) {
            & $python -m pip install --disable-pip-version-check -r pipeline/requirements.txt
            if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned source-intake dependencies." }
            Set-Content -Path $hashFile -Value $requirementsHash -NoNewline
        }

        $command = if ($DryRun) { "inspect" } else { "run" }
        & $python pipeline/source_intake.py $command --root $AgentRoot
        if ($LASTEXITCODE -ne 0) { throw "Source intake failed." }
    }
    finally {
        Pop-Location
    }
} *>&1 | Tee-Object -FilePath $log
