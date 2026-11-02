param([int]$BackendPort = 5000, [int]$FrontendPort = 8501)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:BACKEND_URL = "http://127.0.0.1:$BackendPort"
$env:PORT = "$BackendPort"
$python = (Get-Command python).Source
Start-Process -FilePath $python -ArgumentList '-m', 'backend.app' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'backend.log' -RedirectStandardError 'backend-error.log'
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $health = Invoke-RestMethod "$env:BACKEND_URL/health"
        if ($health.healthy) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw 'Backend did not start. See backend-error.log.' }
Start-Process -FilePath $python -ArgumentList '-m', 'streamlit', 'run', 'streamlit_app.py', '--server.port', "$FrontendPort", '--server.headless', 'true', '--browser.gatherUsageStats', 'false' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'frontend.log' -RedirectStandardError 'frontend-error.log'
Write-Output "Tidal product: http://localhost:$BackendPort/app | Research dashboard: http://localhost:$FrontendPort | API: $env:BACKEND_URL"
