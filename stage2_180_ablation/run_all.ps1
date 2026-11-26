param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [string]$WorkDir = (Join-Path $PSScriptRoot 'scratch'),
    [int]$Workers = 4
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$runner = Join-Path $PSScriptRoot 'scripts\run_experiments.py'
& $PythonExe $runner --phase baseline --data-dir $DataDir --work-dir $WorkDir --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Baseline phase failed; errors are retained in logs.' }
& $PythonExe $runner --phase ablations --data-dir $DataDir --work-dir $WorkDir --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Ablation phase failed; errors are retained in logs.' }
& $PythonExe $runner --phase timing --data-dir $DataDir --work-dir $WorkDir --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Timing phase failed.' }
& $PythonExe (Join-Path $PSScriptRoot 'scripts\analyze_results.py')
if ($LASTEXITCODE -ne 0) { throw 'Result validation failed.' }
& $PythonExe (Join-Path $PSScriptRoot 'scripts\make_figures_tables.py')
if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed.' }
& $PythonExe (Join-Path $PSScriptRoot 'scripts\write_result_notes.py')
if ($LASTEXITCODE -ne 0) { throw 'Result-note generation failed.' }
Write-Host 'Review the PDF/PNG figures. After visual review, run scripts\final_validation.py, scripts\write_result_notes.py, then scripts\package_results.py.'
