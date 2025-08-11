param(
  [Parameter(Position=0)]
  [ValidateSet("ui","ask","ner")]
  [string]$mode = "ui",
  [string]$pdf = ".\data\samples\sample_contract.pdf",
  [string]$q = "",
  [int]$k = 4
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -m venv .venv
  } else {
    & python -m venv .venv
  }
  $py = ".\.venv\Scripts\python.exe"
}
& $py -m pip install --upgrade pip setuptools wheel
& $py -m pip install -r requirements.txt
try { & $py -c "import spacy; spacy.load('en_core_web_sm')" } catch { & $py -m spacy download en_core_web_sm }

if ($mode -ne "ui" -and -not (Test-Path $pdf)) {
  Write-Error "PDF not found: $pdf"
  exit 1
}
switch ($mode) {
  "ui" {
    if (Test-Path $pdf) {
      & $py -m src.app.cli ingest --pdf $pdf
    }
    & $py -m streamlit run .\src\app\streamlit_app.py
  }
  "ask" {
    & $py -m src.app.cli ingest --pdf $pdf
    if (-not $q) { $q = Read-Host "Question" }
    & $py -m src.app.cli ask -q $q --topk $k
  }
  "ner" {
    & $py -m src.app.cli ner --pdf $pdf
  }
}