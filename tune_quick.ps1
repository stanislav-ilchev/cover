# Quick Parameter Tuning - shorter runs to compare convergence
$coverExe = ".\cover.exe"
$baseParams = "v=27 k=6 m=4 t=3 b=86 TC=1 EL=0 verbose=1"

# Quick test configs (frozen=1000000 for ~30sec runs)
$configs = @(
    @("CF=0.99_IT=0.5",     "0.99",    "0.5"),
    @("CF=0.999_IT=0.5",    "0.999",   "0.5"),
    @("CF=0.9999_IT=0.5",   "0.9999",  "0.5"),
    @("CF=0.9999_IT=0.6",   "0.9999",  "0.6"),
    @("CF=0.9999_IT=0.3",   "0.9999",  "0.3"),
    @("CF=1.0_IT=0.6",      "1.0",     "0.6"),
    @("CF=1.0_IT=0.3",      "1.0",     "0.3")
)

Write-Host "Quick Parameter Tuning (frozen=1000000)" -ForegroundColor Cyan
Write-Host ""

$results = @()

foreach ($config in $configs) {
    $name = $config[0]
    $cf = $config[1]
    $it = $config[2]
    
    Write-Host "Testing: $name" -NoNewline
    
    $startTime = Get-Date
    $output = & cmd /c "$coverExe $baseParams CF=$cf IT=$it frozen=1000000" 2>&1
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    # Parse final cost from output
    $finalCost = "N/A"
    $initCost = "N/A"
    foreach ($line in $output) {
        if ($line -match "initCost\s*=\s*(\d+)") { $initCost = $Matches[1] }
        if ($line -match "finalCost\s*=\s*(\d+)") { $finalCost = $Matches[1] }
    }
    
    Write-Host " -> Init: $initCost, Final: $finalCost, Time: $([math]::Round($elapsed,1))s"
    
    $results += [PSCustomObject]@{
        Config = $name
        InitCost = $initCost
        FinalCost = $finalCost
        Time = [math]::Round($elapsed, 1)
    }
}

Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
$results | Sort-Object { [int]$_.FinalCost } | Format-Table -AutoSize



