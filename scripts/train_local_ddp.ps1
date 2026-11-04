# ============================================================
# UNI Cell Instance Segmentation — local DDP training on Windows
# ============================================================
# Runs: UNI access check -> Phase 1 (decoder) -> Phase 2 (LoRA) -> test-set evaluation
# (each best checkpoint with the reference HoVer-Net post-processing, then with the
# notebook's), each training phase under torchrun/DistributedDataParallel (gloo backend).
#
#   powershell -ExecutionPolicy Bypass -File scripts\train_local_ddp.ps1             # 1 process per GPU
#   powershell -ExecutionPolicy Bypass -File scripts\train_local_ddp.ps1 -NProc 1    # what was run on the 1-GPU workstation
#   powershell -ExecutionPolicy Bypass -File scripts\train_local_ddp.ps1 -NProc 2    # 2 ranks; on one WDDM GPU this crashed
#                                                                                    # repeatedly
#   ... -NProc 1 -Backbone imagenet -RunName ablation_imagenet   # same recipe with the ImageNet ViT-L/16 (ablation)
#
# Global batch stays 16 for any -NProc (world x micro-batch x grad-accum), so results
# are comparable across process counts. Re-running resumes from checkpoints\latest_phase*.pth
# (e.g. -SkipPhase1 to continue an interrupted Phase 2). The screen is held on while it runs.
param(
    [int]$NProc = 0,              # 0 = one process per visible GPU
    [int]$Phase1Epochs = 100,
    [int]$Phase2Epochs = 50,
    [int]$NumWorkers = 4,         # DataLoader workers per process
    [string]$UniWeights = '',     # optional local pytorch_model.bin
    [int]$MaxRetries = 3,         # re-launch a crashed training phase; it resumes from latest_phase*.pth
    [ValidateSet('uni', 'imagenet')]
    [string]$Backbone = 'uni',    # 'imagenet' = ablation with timm's ImageNet ViT-L/16, same recipe
    [string]$RunName = '',        # write to checkpoints\<RunName>, logs\<RunName>, results\<RunName>
    [switch]$AllowStandby,        # do not hold the screen on (see the standby guard below)
    [switch]$SkipPhase1,
    [switch]$SkipPhase2,
    [switch]$SkipEval
)

$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir
$Py = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Python venv not found at $Py (see README.md)" }
$env:NO_ALBUMENTATIONS_UPDATE = '1'
$SaveDir, $LogDir, $ResultsDir = 'checkpoints', 'logs', 'results'
if ($RunName) { $SaveDir, $LogDir, $ResultsDir = "checkpoints\$RunName", "logs\$RunName", "results\$RunName" }
New-Item -ItemType Directory -Force -Path $LogDir, $SaveDir, $ResultsDir | Out-Null

if ($NProc -le 0) {
    $NProc = [int](& $Py -c "import torch; print(max(1, torch.cuda.device_count()))")
}
$UniArgs = @()
if ($UniWeights -and $Backbone -eq 'uni') { $UniArgs = @('--uni-weights', $UniWeights) }
$Tag = $Backbone   # evaluation tags: <backbone>_phase1, <backbone>_phase2_lora (+ _notebook_pp)
$Dirs = @('--save-dir', $SaveDir, '--log-dir', $LogDir, '--results-dir', $ResultsDir)

function Invoke-Step([string]$Name, [string[]]$StepArgs) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $start = Get-Date
    & $Py @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit code $LASTEXITCODE)" }
    Write-Host ("=== {0} finished in {1:N1} min ===" -f $Name, ((Get-Date) - $start).TotalMinutes) -ForegroundColor Cyan
}

# Several processes time-slicing one WDDM GPU can hit a driver context-switch fault
# (nvlddmkm event 153) that kills a rank. Such faults are transient, so a failed
# training phase is re-launched and train.py resumes from checkpoints\latest_phase*.pth.
function Stop-OrphanedWorkers {
    # A worker whose torchrun agent is gone (e.g. wedged in a collective) still holds a GPU context.
    $procs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'")
    $alive = @{}
    foreach ($p in $procs) { $alive[$p.ProcessId] = $true }
    foreach ($p in $procs) {
        if ($p.CommandLine -match 'train\.py.*--phase' -and $p.CommandLine -notmatch 'torchrun_compat' -and
            -not $alive.ContainsKey($p.ParentProcessId)) {
            Write-Warning "Stopping orphaned training worker PID $($p.ProcessId)"
            & taskkill.exe /T /F /PID $p.ProcessId | Out-Null
        }
    }
}

function Invoke-Training([string]$Name, [string[]]$StepArgs) {
    for ($attempt = 1; ; $attempt++) {
        try { Invoke-Step $Name $StepArgs; return }
        catch {
            if ($attempt -gt $MaxRetries) { throw }
            Write-Warning "$_ -- retry $attempt of $MaxRetries in 30 s (resumes from the latest checkpoint)"
            Stop-OrphanedWorkers
            Start-Sleep -Seconds 30
        }
    }
}

# Standby guard. On PCs with Modern Standby (S0 low-power idle) the screen-off idle timer
# puts Windows into standby even when "Sleep" is set to Never; a driver that cannot power
# down while the GPU is busy then crashes the machine (bugcheck 0x9F, see
# README.md). ES_DISPLAY_REQUIRED keeps resetting the display idle timer
# for as long as this script runs; no power setting is changed, and the request ends with
# the script. The power button or Start > Sleep still put the PC to sleep.
$ES_CONTINUOUS = [uint32]2147483648                        # 0x80000000
$ES_GUARD = [uint32]2147483651                             # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
if (-not $AllowStandby) {
    if (-not ('UniSeg.Power' -as [type])) {
        Add-Type -Namespace UniSeg -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
    }
    if ([UniSeg.Power]::SetThreadExecutionState($ES_GUARD) -eq 0) {
        Write-Warning 'Could not ask Windows to keep the screen on; an idle standby may interrupt training.'
    } else {
        Write-Host 'Standby guard on: the screen stays on until this script ends (-AllowStandby to disable).'
    }
}

try {
    Write-Host "Project: $ProjectDir | DDP processes: $NProc | backbone: $Backbone | output: $SaveDir, $LogDir, $ResultsDir"
    if ($Backbone -eq 'uni') { Invoke-Step 'UNI access check' (@('scripts\check_uni_access.py') + $UniArgs) }

    $Launch = @('scripts\torchrun_compat.py', '--standalone', "--nproc_per_node=$NProc", 'train.py', '--backbone', $Backbone) + $Dirs
    if (-not $SkipPhase1) {
        Invoke-Training "Phase 1: decoder only ($Backbone frozen)" ($Launch + @('--phase', '1', '--epochs', "$Phase1Epochs",
            '--batch-size', '16', '--lr', '1e-3', '--num-workers', "$NumWorkers") + $UniArgs)
    }
    if (-not $SkipPhase2) {
        Invoke-Training 'Phase 2: LoRA fine-tuning' ($Launch + @('--phase', '2', '--epochs', "$Phase2Epochs",
            '--batch-size', '16', '--lr-lora', '1e-5', '--lr-decoder', '1e-4', '--num-workers', "$NumWorkers") + $UniArgs)
    }
    if (-not $SkipEval) {
        $Eval = @('evaluate.py', '--results-dir', $ResultsDir)
        Invoke-Step 'Evaluate Phase 1' ($Eval + @('--checkpoint', "$SaveDir\best_phase1.pth", '--tag', "${Tag}_phase1") + $UniArgs)
        Invoke-Step 'Evaluate Phase 2' ($Eval + @('--checkpoint', "$SaveDir\best_phase2.pth", '--tag', "${Tag}_phase2_lora") + $UniArgs)
        # Same checkpoints with the notebook's post-processing, for comparison with v1 numbers
        Invoke-Step 'Evaluate Phase 1 (notebook post-processing)' ($Eval + @('--checkpoint', "$SaveDir\best_phase1.pth",
            '--tag', "${Tag}_phase1_notebook_pp", '--postproc', 'notebook', '--num-vis', '0') + $UniArgs)
        Invoke-Step 'Evaluate Phase 2 (notebook post-processing)' ($Eval + @('--checkpoint', "$SaveDir\best_phase2.pth",
            '--tag', "${Tag}_phase2_lora_notebook_pp", '--postproc', 'notebook', '--num-vis', '0') + $UniArgs)
    }
    Write-Host "`nDone. Logs: $LogDir\  Checkpoints: $SaveDir\  Metrics/figures: $ResultsDir\" -ForegroundColor Green
} finally {
    if (-not $AllowStandby) { [UniSeg.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null }
}
