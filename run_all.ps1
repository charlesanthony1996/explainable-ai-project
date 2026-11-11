# Run the whole project end to end, on Windows.
#
#   .\run_all.ps1             everything that needs no language model
#   .\run_all.ps1 -WithLlm    everything, including the Ollama stages
#
# The training pathway takes roughly 30 to 60 minutes. The LLM part adds
# about 20 minutes and needs Ollama running with the model pulled.
#
# Stops at the first failure rather than carrying on with stale inputs: a
# later stage reading half-written models produces numbers that look fine
# and are wrong.

param([switch]$WithLlm)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# uv installs to %USERPROFILE%\.local\bin. A terminal opened before the
# install still has the old PATH, which is the usual reason for
# "uv is not recognized".
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "uv not found. Install it from https://docs.astral.sh/uv/ and reopen the terminal."
    exit 1
}

$started = Get-Date

function Invoke-Stage {
    param([string[]]$Arguments = @())
    Write-Output ""
    Write-Output "=================================================================="
    Write-Output (">>> uv run main.py " + ($Arguments -join " "))
    Write-Output "=================================================================="
    & uv run main.py @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Output ("Stage failed: " + ($Arguments -join " "))
        exit $LASTEXITCODE
    }
}

# --- training: steps 1 to 35 in one command ------------------------------
Invoke-Stage

# --- governance and verification, seconds each ---------------------------
Invoke-Stage verify_fairness  # parity with bootstrap intervals
Invoke-Stage zones            # the four control zones
Invoke-Stage provenance       # provenance record and governance decision
Invoke-Stage overview         # the three overview diagrams

# --- anything that talks to the language model ---------------------------
if ($WithLlm) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 -UseBasicParsing | Out-Null
    } catch {
        Write-Output ""
        Write-Output "Ollama is not answering on 127.0.0.1:11434."
        Write-Output "Start it, then run:  .\run_all.ps1 -WithLlm"
        exit 1
    }
    Invoke-Stage ask       # one applicant, one checked answer
    Invoke-Stage evaluate  # the 45 questions through runs A, B and C
} else {
    Write-Output ""
    Write-Output "Skipped the stages that need Ollama. For those:  .\run_all.ps1 -WithLlm"
}

$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes)
Write-Output ""
Write-Output "=================================================================="
Write-Output "Finished in about $minutes minutes."
Write-Output "  results/tables/   result tables"
Write-Output "  results/figures/  figures"
Write-Output "  docs/             data quality, datasheet, model card, fairness,"
Write-Output "                    governance decision"
Write-Output "  evaluation/       questions, answers, scoring sheets"
Write-Output ""
Write-Output "Still by hand:  uv run main.py team_rubric   (the blind scoring page)"
Write-Output "=================================================================="
