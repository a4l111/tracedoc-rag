param(
    [string]$PythonPath = ".build-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot $PythonPath
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Build Python not found: $Python. Create .build-venv and install requirements.txt plus requirements-build.txt first."
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --noconfirm --clean TraceDocRAG.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $Release = (Resolve-Path "dist\TraceDocRAG").Path
    foreach ($Name in @("config.yaml", ".env.example", "README.md", "DISTRIBUTION.md")) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination $Release -Force
    }
    foreach ($Relative in @("data\raw", "data\parsed", "data\indexes", "data\models", "data\service")) {
        New-Item -ItemType Directory -Path (Join-Path $Release $Relative) -Force | Out-Null
    }

    $Exe = Join-Path $Release "TraceDocRAG.exe"
    $Hash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
    $Architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    $BuildInfo = Join-Path $Release "BUILD_INFO.txt"
    $Lines = @(
        "TraceDoc RAG Windows onedir build",
        "built_at=$([DateTimeOffset]::Now.ToString('o'))",
        "python=$(& $Python --version 2>&1)",
        "architecture=$Architecture",
        "file_count=pending",
        "size_bytes=pending",
        "exe_sha256=$Hash"
    )
    Set-Content -LiteralPath $BuildInfo -Value $Lines -Encoding UTF8
    $Files = Get-ChildItem -LiteralPath $Release -File -Recurse
    $Bytes = ($Files | Measure-Object -Property Length -Sum).Sum
    $Lines[4] = "file_count=$($Files.Count)"
    $Lines[5] = "size_bytes=$Bytes"
    Set-Content -LiteralPath $BuildInfo -Value $Lines -Encoding UTF8
    Write-Output "Release: $Release"
    Write-Output "Size MiB: $([Math]::Round($Bytes / 1MB, 1))"
    Write-Output "EXE SHA256: $Hash"
} finally {
    Pop-Location
}
