# Parallel cover search script
# Runs multiple independent SA searches simultaneously
# Usage: .\run_parallel.ps1 -NumJobs 8 -b 85

param(
    [int]$NumJobs = 4,
    [int]$v = 27,
    [int]$k = 6,
    [int]$m = 4,
    [int]$t = 3,
    [int]$b = 85,
    [double]$CF = 0.9999,
    [double]$IT = 0.6,
    [int]$frozen = 10000000,
    [int]$EL = 0
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$coverExe = Join-Path $scriptDir "cover.exe"

Write-Host "Starting $NumJobs parallel cover searches..." -ForegroundColor Cyan
Write-Host "Parameters: v=$v k=$k m=$m t=$t b=$b CF=$CF IT=$IT frozen=$frozen" -ForegroundColor Yellow
Write-Host ""

$jobs = @()
$startTime = Get-Date

for ($i = 1; $i -le $NumJobs; $i++) {
    $seed = (Get-Random -Maximum 2147483647)
    $logFile = "cover_job$i.log"
    $resFile = "cover_job$i.res"
    
    # Vary InitTemp slightly for each job to explore different regions
    $jobIT = $IT + ($i - 1) * 0.05
    
    $args = "v=$v k=$k m=$m t=$t b=$b CF=$CF IT=$jobIT frozen=$frozen EL=$EL PRNG=$seed log=$logFile result=$resFile verbose=1"
    
    Write-Host "Starting job $i with IT=$jobIT, seed=$seed" -ForegroundColor Green
    
    $job = Start-Process -FilePath $coverExe -ArgumentList $args -PassThru -NoNewWindow -RedirectStandardOutput "job$i.out"
    $jobs += @{Process=$job; Id=$i; IT=$jobIT}
}

Write-Host ""
Write-Host "All jobs started. Monitoring for solutions..." -ForegroundColor Cyan
Write-Host "(Press Ctrl+C to stop all jobs)" -ForegroundColor Gray
Write-Host ""

# Monitor jobs
$foundSolution = $false
while ($jobs.Process | Where-Object { !$_.HasExited }) {
    foreach ($job in $jobs) {
        $resFile = "cover_job$($job.Id).res"
        if ((Test-Path $resFile) -and (Get-Item $resFile).Length -gt 0) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host ""
            Write-Host "*** SOLUTION FOUND by job $($job.Id) in $([math]::Round($elapsed, 1)) seconds! ***" -ForegroundColor Green
            Write-Host "IT=$($job.IT)" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Solution saved to: $resFile" -ForegroundColor Cyan
            Get-Content $resFile | Select-Object -First 10
            if ((Get-Content $resFile).Count -gt 10) {
                Write-Host "... ($(((Get-Content $resFile).Count) - 10) more lines)"
            }
            $foundSolution = $true
            
            # Kill other jobs
            Write-Host ""
            Write-Host "Stopping other jobs..." -ForegroundColor Yellow
            foreach ($otherJob in $jobs) {
                if (!$otherJob.Process.HasExited) {
                    $otherJob.Process.Kill()
                }
            }
            break
        }
    }
    if ($foundSolution) { break }
    Start-Sleep -Milliseconds 500
}

$elapsed = ((Get-Date) - $startTime).TotalSeconds
Write-Host ""
Write-Host "Total time: $([math]::Round($elapsed, 1)) seconds" -ForegroundColor Cyan

if (!$foundSolution) {
    Write-Host "No solution found (all jobs finished without reaching EndLimit)" -ForegroundColor Yellow
}

