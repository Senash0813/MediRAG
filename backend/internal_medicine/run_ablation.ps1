param(
    [string]$InputPath = ".\data\ablation_examples.json",
    [string]$OutputDir = ".\ablation_results",
    [string[]]$Variants = @("baseline", "no_domain_gate", "rag_only", "no_ner", "no_nli", "strict_retrieval", "loose_retrieval"),
    [int]$TopK = 0,
    [double]$Temperature = 0.0,
    [switch]$NoVerify,
    [switch]$ForceReindex
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$pythonArgs = @(
    "-m", "app.scripts.ablation_study",
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--temperature", $Temperature
)

if ($TopK -gt 0) {
    $pythonArgs += @("--top-k", $TopK)
}

if ($Variants.Count -gt 0) {
    $pythonArgs += "--variants"
    $pythonArgs += $Variants
}

if ($NoVerify) {
    $pythonArgs += "--no-verify"
}

if ($ForceReindex) {
    $pythonArgs += "--force-reindex"
}

& $python @pythonArgs