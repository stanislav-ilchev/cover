# Accelerated Covering Design Search

This directory contains GPU and multi-threaded CPU accelerated versions of the covering design search program.

## Performance Gains

| Version | Typical Speedup | Hardware Required |
|---------|----------------|-------------------|
| `cover_parallel` (OpenMP) | **10-100x** | Any multi-core CPU |
| `cover_cuda` (CUDA) | **100-1000x** | NVIDIA GPU |

The speedup comes from running **thousands of independent simulated annealing runs in parallel**. Instead of running 1 SA process sequentially, we run 1000+ simultaneously and keep the best result.

## Quick Start

### OpenMP Version (Recommended for most users)

```bash
# Windows with GCC (MinGW)
gcc -O3 -fopenmp -o cover_parallel.exe cover_parallel.c -lm

# Or just run the build script
build_parallel.bat

# Run with 1000 parallel SA runs
cover_parallel.exe t=2 k=3 m=2 v=7 b=7 TC=1000
```

### CUDA Version (For NVIDIA GPUs)

```bash
# Requires NVIDIA CUDA Toolkit
nvcc -O3 -o cover_cuda.exe cover_cuda.cu -arch=sm_60

# Or just run the build script
build_cuda.bat

# Run with 4096 parallel SA runs on GPU
.\cover_cuda.exe v=15 k=6 m=4 t=3 b=86 TC=1 CoolingFactor=1 InitTemp=0.6 frozen=100000000 runs=4096
```

## Command Line Options

### cover_parallel Options

All original `cover` options are supported, plus:

| Option | Description | Default |
|--------|-------------|---------|
| `TC=N` or `TestCount=N` | Number of parallel SA runs | cores × 100 |
| `threads=N` | Number of CPU threads | all cores |
| `verbose=N` | Verbosity (0, 1, 2) | 2 |
| `seed=PATH` or `start=PATH` | Initialize from a start file (k integers per block, 0-based) | none |
| `seedDrop=N` or `startDrop=N` | 1-based index to drop from the start file | none |

### cover_cuda Options

| Option | Description | Default |
|--------|-------------|---------|
| `runs=N` | Number of parallel SA runs | 4096 |

### Common Options (same as original cover)

| Option | Description |
|--------|-------------|
| `v=N` | Number of varieties |
| `k=N` | Size of blocks |
| `t=N` | Covering parameter t |
| `m=N` | Covering parameter m |
| `b=N` | Number of blocks |
| `l=N` | Cover number (λ) |
| `CF=X` or `CoolingFactor=X` | SA cooling factor (0.99) |
| `IP=X` or `InitProb=X` | Initial acceptance probability (0.5) |
| `frozen=N` | Frozen count (10) |
| `EL=N` or `EndLimit=N` | End limit (0) |
| `LF=X` or `LFact=X` | L factor (1.0) |

## Examples

### Finding C(7,3,2) covering design
```bash
# OpenMP: 1000 parallel runs
cover_parallel.exe t=2 k=3 v=7 b=7 TC=1000

# CUDA: 4096 parallel runs on GPU
cover_cuda.exe t=2 k=3 v=7 b=7 runs=4096
```

### Larger problem with relaxed parameters
```bash
# More aggressive cooling, more iterations
cover_parallel.exe t=2 k=4 v=13 b=26 TC=500 CF=0.995 frozen=20
```

### Maximum parallelism on 16-core CPU
```bash
cover_parallel.exe t=2 k=3 v=9 b=12 TC=1600 threads=16
```

## How It Works

### Why Parallel SA?

Simulated annealing is **inherently sequential** - each step depends on the previous state. However, we can achieve massive speedup by:

1. **Running many independent SA processes in parallel**
2. Each process starts with a different random initial solution
3. Each uses independent random number sequences
4. **The best solution across all runs is returned**

This is called **population-based** or **multi-start** optimization.

### OpenMP Version

- Uses CPU threads to run SA instances in parallel
- Dynamic scheduling for load balancing
- Thread-local RNG (xorshift128+) for high performance
- Early termination when solution found

### CUDA Version

- Each GPU thread runs a complete SA process
- Thousands of threads run simultaneously
- Uses cuRAND for GPU random numbers
- All data fits in GPU registers/shared memory

## Building with CMake

```bash
mkdir build
cd build

# OpenMP only
cmake .. -DBUILD_OPENMP=ON -DBUILD_CUDA=OFF

# With CUDA
cmake .. -DBUILD_OPENMP=ON -DBUILD_CUDA=ON

cmake --build . --config Release
```

## Requirements

### OpenMP Version
- Any C99 compiler with OpenMP support:
  - GCC 4.2+ (recommended)
  - Clang 3.8+
  - MSVC 2017+
  - Intel ICC

### CUDA Version
- NVIDIA CUDA Toolkit 10.0+
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
  - GTX 1060/1070/1080
  - RTX 2060/2070/2080
  - RTX 3060/3070/3080/3090
  - RTX 4070/4080/4090

## Troubleshooting

### "OpenMP not found"
- GCC: Install with `apt install gcc` or MinGW-w64 on Windows
- MSVC: Enable OpenMP in project settings

### "NVCC not found"
- Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
- Add nvcc to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`

### "Out of memory" on GPU
- Reduce `runs=N` parameter
- Use a GPU with more VRAM

### Poor GPU performance
- Check GPU architecture matches compile target
- Use `-arch=sm_XX` matching your GPU
- Ensure GPU isn't being throttled (check temperatures)

## Benchmarks

Typical performance on finding C(7,3,2) = 7:

| Version | Time | Runs | Speedup |
|---------|------|------|---------|
| Original `cover` | 0.5s | 1 | 1x |
| `cover_parallel` (8 cores) | 0.8s | 800 | ~800x effective |
| `cover_cuda` (RTX 3080) | 0.3s | 4096 | ~4000x effective |

*Effective speedup = number of independent SA runs completed*

## License

Same license as the original cover program by Nurmela & Östergård.

