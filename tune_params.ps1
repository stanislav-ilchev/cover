# Parameter Tuning Script for cover.exe
# Tests different SA parameter combinations and reports best settings

$coverExe = ".\cover.exe"
$baseParams = "v=27 k=6 m=4 t=3 b=86 TC=1 EL=0 verbose=0"

# Test configurations - shorter runs to compare convergence speed
$configs = @(
    # Format: Name, CF, IT, frozen, LFact
    @("Fast-Cool-HighT",    "0.99",    "0.8",  "10000000", "1"),
    @("Fast-Cool-MedT",     "0.99",    "0.5",  "10000000", "1"),
    @("Fast-Cool-LowT",     "0.99",    "0.3",  "10000000", "1"),
    @("Med-Cool-HighT",     "0.999",   "0.8",  "10000000", "1"),
    @("Med-Cool-MedT",      "0.999",   "0.5",  "10000000", "1"),
    @("Med-Cool-LowT",      "0.999",   "0.3",  "10000000", "1"),
    @("Slow-Cool-HighT",    "0.9999",  "0.8",  "10000000", "1"),
    @("Slow-Cool-MedT",     "0.9999",  "0.5",  "10000000", "1"),
    @("Slow-Cool-LowT",     "0.9999",  "0.3",  "10000000", "1"),
    @("VSlow-Cool-MedT",    "0.99999", "0.5",  "10000000", "1"),
    @("NoСool-LowT",        "1.0",     "0.3",  "10000000", "1"),
    @("NoCool-VLowT",       "1.0",     "0.1",  "10000000", "1"),
    @("Med-Cool-2xL",       "0.999",   "0.5",  "10000000", "2"),
    @("Slow-Cool-2xL",      "0.9999",  "0.5",  "10000000", "2")
)

Write-Host "========================================"
Write-Host "  Cover.exe Parameter Tuning"
Write-Host "  Testing $($configs.Count) configurations"
Write-Host "========================================"
Write-Host ""

$results = @()

foreach ($config in $configs) {
    $name = $config[0]
    $cf = $config[1]
    $it = $config[2]
    $frozen = $config[3]
    $lf = $config[4]
    
    Write-Host "Testing: $name (CF=$cf IT=$it frozen=$frozen LF=$lf)" -ForegroundColor Cyan
    
    $cmd = "$coverExe $baseParams CF=$cf IT=$it frozen=$frozen LF=$lf"
    
    $startTime = Get-Date
    $output = & cmd /c $cmd 2>&1
    $endTime = Get-Date
    $elapsed = ($endTime - $startTime).TotalSeconds
    
    # Parse output for final cost
    $finalCost = -1
    $iterations = 0
    foreach ($line in $output) {
        if ($line -match "finalCost\s*=\s*(\d+)") {
            $finalCost = [int]$Matches[1]
        }
        if ($line -match "iterations\s*=\s*(\d+)") {
            $iterations = [int64]$Matches[1]
        }
        # Also check for "cost" in other formats
        if ($line -match "cost\s*[:=]\s*(\d+)") {
            $finalCost = [int]$Matches[1]
        }
    }
    
    $iterPerSec = if ($elapsed -gt 0) { [math]::Round($iterations / $elapsed) } else { 0 }
    
    Write-Host "  Result: Cost=$finalCost, Time=$([math]::Round($elapsed,1))s, Iter=$iterations ($iterPerSec/sec)" -ForegroundColor Yellow
    
    $results += [PSCustomObject]@{
        Name = $name
        CF = $cf
        IT = $it
        Frozen = $frozen
        LF = $lf
        FinalCost = $finalCost
        TimeSeconds = [math]::Round($elapsed, 1)
        Iterations = $iterations
        IterPerSec = $iterPerSec
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Results Summary (sorted by cost)"
Write-Host "========================================"

$results | Sort-Object FinalCost, TimeSeconds | Format-Table -AutoSize

Write-Host ""
Write-Host "Best configuration:" -ForegroundColor Green
$best = $results | Sort-Object FinalCost, TimeSeconds | Select-Object -First 1
Write-Host "  $($best.Name): CF=$($best.CF) IT=$($best.IT) frozen=$($best.Frozen) LF=$($best.LF)"
Write-Host "  Final Cost: $($best.FinalCost), Time: $($best.TimeSeconds)s"

# Save results
$results | Export-Csv -Path "tuning_results.csv" -NoTypeInformation
Write-Host ""
Write-Host "Results saved to tuning_results.csv"



