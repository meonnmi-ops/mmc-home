import numpy as np
import subprocess

def run_test(binary, M, K, N, label):
    r = subprocess.run([binary], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{label}] FAILED: {r.stderr}")
        return None, None
    # Parse output
    gflops = None
    ms = None
    for line in r.stdout.splitlines():
        if 'GFLOPS' in line:
            gflops = float(line.split(':')[1].strip())
        if 'ms/run' in line:
            ms = float(line.split(',')[1].split('ms')[0].strip())
    C = np.fromfile('/tmp/C.bin', dtype=np.float32).reshape(M, N)
    return gflops, C

def main():
    M, K, N = 128, 256, 1024
    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np.float32)
    B = (np.random.randn(K, N) * 0.1).astype(np.float32)
    A.tofile('/tmp/A.bin'); B.tofile('/tmp/B.bin')

    C_ref = A @ B
    print("=" * 70)
    print(f"Shape: {M}x{K}x{N}  ({2*M*K*N/1e6:.2f} MFLOP)")
    print("=" * 70)
    print(f"{'Method':<20} {'GFLOPS':>12} {'Max abs diff':>15} {'Status':>8}")
    print("-" * 70)

    results = {}
    for name, binary in [
        ("Naive CUDA",   "/tmp/test_mm"),
        ("Tiled CUDA",   "/tmp/test_mm_tiled"),
        ("cuBLAS",       "/tmp/test_mm_cublas"),
    ]:
        gflops, C = run_test(binary, M, K, N, name)
        if gflops is None:
            print(f"{name:<20} {'(failed)':>12}")
            continue
        diff = np.abs(C_ref - C).max()
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"{name:<20} {gflops:>10.2f}   {diff:>13.3e}   {status:>8}")
        results[name] = gflops

    print("-" * 70)
    if "Tiled CUDA" in results and "cuBLAS" in results:
        speedup = results["cuBLAS"] / results["Tiled CUDA"]
        print(f"cuBLAS vs Tiled: {speedup:.2f}x faster")
    print("=" * 70)

if __name__ == "__main__":
    main()
