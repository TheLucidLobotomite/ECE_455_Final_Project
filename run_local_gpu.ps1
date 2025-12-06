# PowerShell script to compile and run GPU DFT on local Windows machine
# Make sure you have:
# 1. CUDA Toolkit installed (with nvcc in PATH)
# 2. MSYS2 with FFTW installed
# 3. NVIDIA GPU with compute capability 3.5+

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPU DFT Local Compilation and Execution" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set up paths
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\ucrt64\lib;" + $env:PATH

# Check if CUDA is available
Write-Host "Checking for CUDA..." -ForegroundColor Yellow
$cudaCheck = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $cudaCheck) {
    Write-Host "ERROR: nvcc not found. Please ensure CUDA Toolkit is installed and in PATH." -ForegroundColor Red
    exit 1
}
Write-Host "CUDA found: $($cudaCheck.Source)" -ForegroundColor Green
Write-Host ""

# Compile
Write-Host "Compiling test_dft_gpu.cu..." -ForegroundColor Yellow
nvcc test_dft_gpu.cu -o test_dft_gpu.exe `
    -I"C:\msys64\ucrt64\include" `
    -L"C:\msys64\ucrt64\lib" `
    -lcusolver -lcublas -lfftw3 -lfftw3_threads -lopenblas `
    -Xcompiler "/openmp" `
    --std=c++11 `
    -O3

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Compilation successful!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running DFT Calculation on GPU" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run
.\test_dft_gpu.exe

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Execution completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan