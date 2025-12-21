# Parameter tuning for cover_big_cuda.exe

$ErrorActionPreference = "Stop"

$coverExe = ".\cover_big_cuda.exe"
if (-not (Test-Path $coverExe)) {
    Write-Error "Missing $coverExe. Build it first."
    exit 1
}

# Base parameters (edit as needed)
$v = 49
$k = 6
$m = 6
$t = 3
$b = 160
$runs = 256
$rr = 1
$sample = 0

$startFile = "solution_b160_start.txt"
$startFileIsZeroBased = $null  # $true, $false, or $null for auto-detect

function Get-FileNumbers([string]$path) {
    $numbers = @()
    foreach ($line in Get-Content -Path $path) {
        $trim = $line.Trim()
        if ($trim.Length -eq 0) { continue }
        if ($trim.StartsWith("#")) { continue }
        foreach ($m in [regex]::Matches($line, "\d+")) {
            $numbers += [int]$m.Value
        }
    }
    return $numbers
}

function Get-BlockCount([string]$path) {
    $count = 0
    foreach ($line in Get-Content -Path $path) {
        $trim = $line.Trim()
        if ($trim.Length -eq 0) { continue }
        if ($trim.StartsWith("#")) { continue }
        if ([regex]::IsMatch($line, "\d+")) { $count++ }
    }
    return $count
}

function Convert-StartFileTo1Based([string]$inPath, [string]$outPath) {
    $outLines = @()
    foreach ($line in Get-Content -Path $inPath) {
        $trim = $line.Trim()
        if ($trim.Length -eq 0) { continue }
        if ($trim.StartsWith("#")) {
            $outLines += $line
            continue
        }
        $nums = @()
        foreach ($m in [regex]::Matches($line, "\d+")) {
            $nums += ([int]$m.Value + 1)
        }
        if ($nums.Count -gt 0) {
            $outLines += ($nums -join " ")
        }
    }
    Set-Content -Path $outPath -Value $outLines -Encoding ASCII
}

function Prepare-StartFile([string]$path, [int]$v, [Nullable[bool]]$isZeroBased) {
    if (-not (Test-Path $path)) {
        Write-Warning "Start file not found: $path. Continuing without start file."
        return $null
    }

    $blockCount = Get-BlockCount $path
    if ($blockCount -gt 0 -and $blockCount -ne $b) {
        Write-Warning "Start file has $blockCount blocks; b=$b."
    }

    $numbers = Get-FileNumbers $path
    if ($numbers.Count -eq 0) { return $path }

    $min = ($numbers | Measure-Object -Minimum).Minimum
    $max = ($numbers | Measure-Object -Maximum).Maximum

    $zeroBased = $false
    if ($isZeroBased -ne $null) {
        $zeroBased = [bool]$isZeroBased
    } else {
        $zeroBased = ($min -eq 0 -or $max -le ($v - 1))
    }

    if (-not $zeroBased) { return $path }

    $dir = Split-Path -Parent $path
    $base = [IO.Path]::GetFileNameWithoutExtension($path)
    $ext = [IO.Path]::GetExtension($path)
    $outPath = Join-Path $dir ($base + "_1based" + $ext)

    Convert-StartFileTo1Based -inPath $path -outPath $outPath
    Write-Host "Converted start file to 1-based: $outPath" -ForegroundColor Yellow
    return $outPath
}

$preparedStart = Prepare-StartFile -path $startFile -v $v -isZeroBased $startFileIsZeroBased

$baseArgs = @(
    "v=$v", "k=$k", "m=$m", "t=$t", "b=$b",
    "runs=$runs", "RR=$rr"
)
if ($sample -gt 0) { $baseArgs += "sample=$sample" }
if ($preparedStart) { $baseArgs += "start=$preparedStart" }

$logRoot = Join-Path $PSScriptRoot "tuning_logs"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $logRoot $stamp
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Run-Test([string]$tag, [int]$th, [int]$iter, [int]$rounds, [int]$exact, [int]$parallel) {
    $cliArgs = $baseArgs + @(
        "iter=$iter",
        "rounds=$rounds",
        "TH=$th",
        "exact=$exact",
        "parallel=$parallel"
    )

    $logFile = Join-Path $logDir ("{0}_TH{1}_iter{2}_rounds{3}_exact{4}_parallel{5}.log" -f $tag, $th, $iter, $rounds, $exact, $parallel)
    Write-Host ("Running {0}: TH={1} iter={2} rounds={3} exact={4} parallel={5}" -f $tag, $th, $iter, $rounds, $exact, $parallel) -ForegroundColor Cyan

    $elapsed = Measure-Command {
        & $coverExe @cliArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Null
    }

    $bestCost = -1
    $bestLine = Select-String -Path $logFile -Pattern "bestCost\s*=\s*(\d+)" | Select-Object -Last 1
    if ($bestLine) {
        $bestCost = [int]$bestLine.Matches[0].Groups[1].Value
    }

    $elapsedSeconds = [math]::Round($elapsed.TotalSeconds, 1)
    Write-Host "  Result: bestCost=$bestCost time=${elapsedSeconds}s" -ForegroundColor Yellow

    $solutionPath = Join-Path $PSScriptRoot "solution.txt"
    if (Test-Path $solutionPath) {
        $solCopy = Join-Path $logDir ("solution_{0}_TH{1}_iter{2}_rounds{3}_exact{4}_parallel{5}.txt" -f $tag, $th, $iter, $rounds, $exact, $parallel)
        Copy-Item -Path $solutionPath -Destination $solCopy -Force
    }

    return [PSCustomObject]@{
        Tag = $tag
        TH = $th
        Iter = $iter
        Rounds = $rounds
        Exact = $exact
        Parallel = $parallel
        BestCost = $bestCost
        TimeSeconds = $elapsedSeconds
        LogFile = $logFile
    }
}

# Exact-only sweep (no sampling)
$thValues = @(200, 400, 600, 800, 1200)
$iter = 1000
$rounds = 1
$exact = 1
$parallel = 1

$exactResults = @()
foreach ($th in $thValues) {
    $exactResults += Run-Test -tag "exact" -th $th -iter $iter -rounds $rounds -exact $exact -parallel $parallel
}

$exactCsv = Join-Path $logDir "exact_results.csv"
$exactResults | Export-Csv -Path $exactCsv -NoTypeInformation
Write-Host ""
Write-Host "Exact sweep results (sorted by bestCost):" -ForegroundColor Green
$exactResults | Sort-Object BestCost, TimeSeconds | Format-Table -AutoSize

Write-Host ""
Write-Host "Logs and CSVs saved to: $logDir" -ForegroundColor Green
