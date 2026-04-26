# Advanced PC Verification Suite - 2025/2026 Edition

**Comprehensive PC/GPU validation based on industry best practices from Together AI, NVIDIA DCGM, gpu-burn, and MemtestG80.**

## What's New in This Edition

This is a **complete rewrite** based on extensive research of 2025-2026 best practices:

### Based on Industry Leaders:

1. **Together AI Methodology** (AI infrastructure at scale)
   - Multi-level stress testing
   - Long-duration sustained tests
   - Comprehensive GPU validation

2. **NVIDIA DCGM Diagnostics** (Official data center GPU manager)
   - Level 1-3 test progression
   - Targeted stress with GEMM operations
   - STREAM benchmark memory bandwidth
   - Pattern-based memory error detection

3. **gpu-burn** concepts
   - Maximum VRAM allocation (95%+ capacity)
   - Sustained memory holding
   - Tensor core utilization

4. **MemtestG80** methodology
   - Pattern-based memory testing
   - Error detection algorithms
   - Comprehensive VRAM validation

## Key Features

### 1. Maximum VRAM Testing
Unlike basic tests that allocate 2GB, this fills **95% of your GPU memory** (23+ GB on RTX 3090) and holds it for the duration of the test.

```bash
python pc_verify_advanced.py --vram-fraction 0.95
```

### 2. Pattern-Based Memory Error Detection
Implements MemtestG80-style pattern tests:
- All zeros (0x00000000)
- All ones (0xFFFFFFFF)
- Alternating patterns (0xAAAAAAAA, 0x55555555)
- Random patterns
- Byte patterns

Detects memory errors that simple allocation tests miss.

### 3. STREAM Benchmark Memory Bandwidth
Uses the industry-standard STREAM benchmark TRIAD operation (like DCGM):
```
a(i) = b(i) + q*c(i)
```

Measures sustainable memory bandwidth, not peak bursts.

### 4. GEMM-Based Targeted Stress (DCGM-style)
Uses cuBLAS GEMM operations (matrix multiply) like DCGM's `targeted_stress` plugin:
- Maintains specific GFLOPS targets
- Uses multiple CUDA streams
- Tests compute engine stability

### 5. Tensor Core Stress Testing
Dedicated tests for mixed precision:
- **FP16** (Half precision) - 2x speedup on tensor cores
- **BF16** (Brain float) - same range as FP32, half memory
- Tests Ampere/Ada tensor core paths

### 6. Mixed Precision ML Training
Tests real-world ML scenarios:
- Automatic Mixed Precision (AMP)
- TF32 tensor math
- Batch size scaling (32 → 1024)
- Throughput measurement

### 7. Large Batch OOM Testing
Finds your GPU's actual memory limits:
- Tests up to batch size 8192
- Determines maximum before OOM
- Useful for model sizing

### 8. Extended GPU Burn
Three-phase burn-in test:
1. GEMM compute stress
2. Memory bandwidth saturation
3. Mixed precision compute

## Test Levels

```bash
# Quick check (2-3 minutes)
python pc_verify_advanced.py --level QUICK --duration 60

# Medium testing (10-15 minutes)
python pc_verify_advanced.py --level MEDIUM --duration 180

# Extensive testing (30-45 minutes) - DEFAULT
python pc_verify_advanced.py --level EXTENSIVE --duration 300

# Maximum testing (1-2 hours)
   python pc_verify_advanced.py --level MAXIMUM --duration 600
```

## Usage Examples

### Standard Test (Recommended for PC validation)
```bash
python pc_verify_advanced.py --duration 300 --vram-fraction 0.95
```

### Quick Sanity Check
```bash
python pc_verify_advanced.py --level QUICK --duration 60
```

### Maximum Intensity (Find thermal issues)
```bash
python pc_verify_advanced.py --level MAXIMUM --duration 600 --vram-fraction 0.98
```

### Custom Report File
```bash
python pc_verify_advanced.py --output my_pc_report.json
```

## Test Output

### Console Output
- Real-time progress bars
- Color-coded results (Green=Pass, Yellow=Warn, Red=Fail)
- Temperature monitoring
- Throughput metrics

### JSON Report
Saved to `verification_report_advanced.json`:
```json
{
  "timestamp": "2026-04-21T...",
  "test_level": "EXTENSIVE",
  "system": {
    "gpu": "NVIDIA GeForce RTX 3090",
    "gpu_memory_gb": 24.0,
    ...
  },
  "results": [...],
  "summary": {
    "total_tests": 14,
    "passed": 14,
    "warnings": 0,
    "failed": 0
  }
}
```

## What Each Test Does

| Test | Duration | What It Tests | Based On |
|------|----------|---------------|----------|
| System Info | <1s | Hardware detection | Standard |
| CPU Stress | 5min | All cores at 100% | Standard |
| Memory Stress | 5min | RAM allocation to 85%+ | Standard |
| Disk I/O | ~2min | Sequential R/W 2GB | Standard |
| GPU Memory Bandwidth (STREAM) | 2min | STREAM TRIAD benchmark | DCGM |
| GPU Memory Pattern Test | 2min | Pattern-based error detection | MemtestG80 |
| GPU Maximum VRAM | 5min | 95% allocation + hold | gpu-burn |
| GPU GEMM Stress | 5min | cuBLAS matrix operations | DCGM targeted_stress |
| GPU Tensor Core FP16 | 5min | Half precision tensor cores | MLPerf |
| GPU Tensor Core BF16 | 5min | Brain float tensor cores | MLPerf |
| ML Training Mixed Precision | 5min | AMP/TF32 training | PyTorch best practice |
| ML Large Batch OOM | 2min | Maximum batch size discovery | AI Research |
| Extended GPU Burn | 5min | Three-phase burn-in | gpu-burn + DCGM |

## Interpreting Results

### PASS
Test completed successfully, no issues detected.

### WARN
Test completed but with concerns:
- High temperatures (>85°C)
- Performance variance >10%
- Memory errors detected

### FAIL
Test failed:
- Hardware errors detected
- Temperature limits exceeded
- Performance below threshold

### SKIP
Test skipped (e.g., BF16 not supported on older GPUs).

## What to Look For

### For New PC Validation:
1. **All tests PASS** - System is stable
2. **Temperatures <85°C** - Cooling adequate
3. **No memory errors** - VRAM is good
4. **Consistent performance** - No thermal throttling

### For AI Research:
1. **Tensor Core tests pass** - Mixed precision works
2. **High ML throughput** - 20,000+ samples/sec on RTX 3090
3. **Large batch OOM >2048** - Good for LLM training
4. **Memory bandwidth >700 GB/s** - RTX 3090 expected

## Troubleshooting

### Out of Memory Errors
- Reduce `--vram-fraction` to 0.90 or 0.85
- Run with `--level MEDIUM`

### Temperature Too High
- Ensure case airflow
- Increase duration for thermal testing
- Watch for thermal throttling in results

### Test Failures
- Check GPU drivers
- Verify CUDA installation
- Look at `verification_report_advanced.json` for details

## Comparison with Basic Version

| Feature | Basic (pc_verify.py) | Advanced (pc_verify_advanced.py) |
|---------|---------------------|--------------------------------|
| VRAM Test | 2GB allocation | 95% capacity (23GB on 3090) |
| Memory Patterns | No | Yes (MemtestG80-style) |
| STREAM Benchmark | No | Yes (DCGM-style) |
| GEMM Stress | Basic | DCGM targeted_stress |
| Tensor Cores | No | FP16 + BF16 tests |
| Test Duration | ~10 min | ~45 min (default) |
| Test Levels | 1 | 4 (QUICK to MAXIMUM) |
| ML Throughput | Basic | AMP/TF32 optimized |
| Large Batch OOM | No | Yes |
| JSON Report | Basic | Comprehensive with warnings |

## References

- Together AI GPU Testing: https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models
- NVIDIA DCGM: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
- gpu-burn: https://github.com/wilicc/gpu-burn
- MemtestG80: https://github.com/ihaque/memtestG80
- STREAM Benchmark: https://www.cs.virginia.edu/stream/

## Requirements

- Python 3.8+
- PyTorch 2.5+ with CUDA support
- nvidia-ml-py (for NVML access)
- psutil, tqdm, tabulate, colorama
- NVIDIA GPU with CUDA support

## Installation

```bash
# PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install nvidia-ml-py psutil tqdm tabulate colorama

# Run
python pc_verify_advanced.py --duration 300
```

## For Sellers/Validation Reports

This suite provides comprehensive documentation for PC sellers:
- JSON report with full metrics
- Temperature logs
- Performance benchmarks
- Error detection results

Share `verification_report_advanced.json` as proof of stability.

---

**2025-2026 Edition** - Incorporating latest industry best practices from leading AI infrastructure providers.
