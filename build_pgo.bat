@echo off
REM Build script using Profile-Guided Optimization (PGO)
REM This compiles twice: first to generate profile data, then to optimize based on it

echo ===== Building cover_pgo.exe with Profile-Guided Optimization =====
echo.

REM Try to find GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo GCC not found in PATH, trying MSYS2...
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

echo Step 1: Building instrumented version for profiling...
gcc -O3 -march=native -ffast-math -fprofile-generate -c cover.c -o cover.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c bincoef.c -o bincoef.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c tables.c -o tables.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c setoper.c -o setoper.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c anneal.c -o anneal.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c solcheck.c -o solcheck.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c exp.c -o exp.o
gcc -O3 -march=native -ffast-math -fprofile-generate -c arg.c -o arg.o
gcc -O3 -march=native -fprofile-generate -o cover_profile.exe cover.o bincoef.o tables.o setoper.o anneal.o solcheck.o exp.o arg.o -lm
if %errorlevel% neq 0 goto :error

echo.
echo Step 2: Running profiling workload (this takes a minute)...
.\cover_profile.exe v=19 k=5 m=3 t=2 b=30 CF=0.999 IT=0.6 frozen=100 L=100000 verbose=0
.\cover_profile.exe v=27 k=6 m=4 t=3 b=100 CF=0.99 IT=0.6 frozen=20 L=20000 verbose=0

echo.
echo Step 3: Building optimized version using profile data...
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c cover.c -o cover.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c bincoef.c -o bincoef.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c tables.c -o tables.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c setoper.c -o setoper.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c anneal.c -o anneal.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c solcheck.c -o solcheck.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c exp.c -o exp.o
gcc -O3 -march=native -ffast-math -flto -fprofile-use -c arg.c -o arg.o
gcc -O3 -march=native -flto -fprofile-use -o cover_pgo.exe cover.o bincoef.o tables.o setoper.o anneal.o solcheck.o exp.o arg.o -lm
if %errorlevel% neq 0 goto :error

echo.
echo ===== SUCCESS: cover_pgo.exe built with PGO! =====
echo.
echo Profile-guided optimization typically gives 10-30%% speedup
echo by optimizing branch prediction and function inlining.
echo.

REM Cleanup profile data
del /q *.gcda 2>nul
del /q cover_profile.exe 2>nul

goto :end

:error
echo.
echo BUILD FAILED!

:end

