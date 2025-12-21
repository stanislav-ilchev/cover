@echo off
REM Build script for cover_parallel (OpenMP version)
REM This version works on any multi-core CPU and provides significant speedup

echo.
echo Building cover_parallel (OpenMP multi-threaded version)
echo ========================================================
echo.

REM Try MSYS2/MinGW64 first (common location)
if exist "C:\msys64\mingw64\bin\gcc.exe" (
    echo Found MinGW64 at C:\msys64\mingw64\bin - compiling with OpenMP...
    "C:\msys64\mingw64\bin\gcc.exe" -O3 -fopenmp -o cover_parallel.exe cover_parallel.c -lm
    if %errorlevel% == 0 (
        echo.
        echo SUCCESS! Built cover_parallel.exe
        echo.
        echo Usage: cover_parallel.exe t=2 k=3 m=2 v=7 b=7 TC=1000
        echo.
        echo Options:
        echo   TC=N        - Number of parallel SA runs (default: cores x 100)
        echo   threads=N   - Number of CPU threads to use
        echo   All other options same as original cover program
        goto :end
    )
)

REM Try GCC in PATH
where gcc >nul 2>nul
if %errorlevel% == 0 (
    echo Found GCC in PATH - compiling with OpenMP...
    gcc -O3 -fopenmp -o cover_parallel.exe cover_parallel.c -lm
    if %errorlevel% == 0 (
        echo.
        echo SUCCESS! Built cover_parallel.exe
        echo.
        echo Usage: cover_parallel.exe t=2 k=3 m=2 v=7 b=7 TC=1000
        echo.
        echo Options:
        echo   TC=N        - Number of parallel SA runs (default: cores x 100)
        echo   threads=N   - Number of CPU threads to use
        echo   All other options same as original cover program
        goto :end
    )
)

REM Try MSVC
where cl >nul 2>nul
if %errorlevel% == 0 (
    echo Found MSVC - compiling with OpenMP...
    cl /O2 /openmp cover_parallel.c /Fe:cover_parallel.exe
    if %errorlevel% == 0 (
        echo.
        echo SUCCESS! Built cover_parallel.exe
        goto :end
    )
)

REM Try Clang
where clang >nul 2>nul
if %errorlevel% == 0 (
    echo Found Clang - compiling with OpenMP...
    clang -O3 -fopenmp -o cover_parallel.exe cover_parallel.c -lm
    if %errorlevel% == 0 (
        echo.
        echo SUCCESS! Built cover_parallel.exe
        goto :end
    )
)

echo.
echo ERROR: No suitable compiler found!
echo.
echo Please install one of:
echo   1. MinGW-w64 (GCC for Windows) - https://www.mingw-w64.org/
echo   2. Visual Studio with C++ workload - https://visualstudio.microsoft.com/
echo   3. LLVM/Clang - https://releases.llvm.org/
echo.

:end
pause

