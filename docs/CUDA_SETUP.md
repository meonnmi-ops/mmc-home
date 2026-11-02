# CUDA Setup — GTX 1660 Ti (WSL2, Debian 13)

## Versions
- CUDA: 12.8 (V12.8.93) at /usr/local/cuda-12.8
- Driver: 615.71.08 (KMD 616.92, CUDA UMD 13.4)
- gcc: 13.3.0 (for nvcc compatibility)

## GLIBC 2.41 fix (Debian 13)
- Compat header: ~/cuda_compat/glibc_c23_math_compat.h
- Env: NVCC_PREPEND_FLAGS="-Xcompiler=-include,$HOME/cuda_compat/glibc_c23_math_compat.h"

## Verify
nvcc -O3 test.cu -o test_cuda && ./test_cuda
# CUDA devices: 1
# NVIDIA GeForce GTX 1660 Ti | Compute 7.5 | 6143 MB

## Hardware
- GTX 1660 Ti: Compute 7.5 (Turing), 24 SMs, 6144 MB VRAM
- CPU: i7-9750H (6 physical cores)
- WSL RAM: 7.7 GB + 8 GB swap = 15.7 GB total
