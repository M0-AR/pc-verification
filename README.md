# PC Verification Suite for AI Research

Comprehensive PC testing tool for AI researchers to validate new hardware with GPU stress testing and ML training simulation.

Based on 2025-2026 industry best practices from Together AI, NVIDIA DCGM, and gpu-burn concepts.

## Features

- **System Information** - Detailed hardware detection
- **CPU Stress Test** - Multi-core intensive computation
- **Memory Stress Test** - RAM allocation and bandwidth testing
- **Disk I/O Test** - Sequential read/write performance
- **GPU Memory Bandwidth** - CUDA memory transfer testing
- **GPU Compute Stress** - Sustained GFLOPS measurement
- **ML Training Simulation** - PyTorch training with synthetic data
- **Mixed Workload** - Combined CPU+GPU+Memory stress

## Requirements

- Python 3.8+ (tested with 3.14.4)
- NVIDIA GPU with CUDA (optional but recommended)
- Windows, Linux, or macOS

## Quick Start

### Windows
```batch
run_tests.bat
```

### Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Run full verification (default 60s per test)
python pc_verify.py

# Run with custom duration
python pc_verify.py --duration 120

# Custom batch sizes for ML training
python pc_verify.py --duration 60 --batch-sizes 32 64 128 256 512
```

## Output

Results are saved to `verification_report.json` with:
- Test status (PASS/SKIP/ERROR)
- Performance scores
- Temperature monitoring
- System resource usage

## What This Tests

### For the Seller/Validation
- GPU stability under sustained load
- Memory errors and overheating
- System throttling issues
- Hardware defects

### For AI Research
- ML training throughput at various batch sizes
- GPU memory bandwidth
- Mixed precision performance
- Multi-GPU scaling
- Realistic training simulation with synthetic data (no downloads needed)

## Test Details

| Test | Duration | Purpose |
|------|----------|---------|
| System Info | <1s | Hardware inventory |
| CPU Stress | 60s | CPU stability & performance |
| Memory Stress | 60s | RAM integrity |
| Disk I/O | ~30s | Storage performance |
| GPU Memory | 60s | Memory bandwidth |
| GPU Compute | 60s | Sustained compute stress |
| ML Training | ~5min | PyTorch training simulation |
| Mixed Workload | 60s | Combined system stress |

## Interpreting Results

- **PASS** - Test completed successfully
- **SKIP** - Test skipped (e.g., no GPU detected)
- **ERROR** - Test failed, possible hardware issue

Check `verification_report.json` for detailed metrics including:
- GPU temperatures (watch for >85°C)
- Memory bandwidth (should match GPU specs)
- Training throughput (samples/sec)
- System resource utilization

## Troubleshooting

**CUDA Out of Memory**: Reduce batch sizes: `--batch-sizes 16 32 64`

**Too slow on CPU**: The tool works without GPU but ML tests will be slow. Consider getting a GPU for AI research.

**Temperature warnings**: Ensure adequate cooling. Stop if GPU exceeds 90°C.

## References

Based on industry best practices:
- [Together AI GPU Cluster Testing](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models)
- [NVIDIA DCGM Diagnostics](https://developer.nvidia.com/dcgm)
- [Civo ML Stress Testing Guide](https://www.civo.com/learn/stress-testing-machine-learning-models)

## License

MIT - Use freely for PC validation.
