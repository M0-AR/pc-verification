#!/usr/bin/env python3
"""
PC Verification Suite for AI Research
Comprehensive testing for new PC with GPU stress testing and ML training simulation.
Based on 2025-2026 industry best practices from Together AI, NVIDIA DCGM, and gpu-burn concepts.
"""

import os
import sys
import time
import json
import argparse
import threading
import psutil
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# GPU monitoring - nvidia-ml-py is the official NVIDIA package
try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    from pynvml import nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage, NVML_TEMPERATURE_GPU
except ImportError:
    try:
        from nvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
        from nvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
        from nvml import nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage, NVML_TEMPERATURE_GPU
    except ImportError:
        nvmlInit = None

# ML / PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# Progress and formatting
from tqdm import tqdm
from colorama import init, Fore, Style
from tabulate import tabulate

init(autoreset=True)


@dataclass
class TestResult:
    """Store results from each test."""
    test_name: str
    status: str
    duration: float
    score: Optional[float] = None
    details: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)


class SyntheticDataset(Dataset):
    """Generate synthetic data for ML training stress tests."""
    
    def __init__(self, num_samples: int = 10000, input_dim: int = 1024, output_dim: int = 256):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Pre-generate random data on CPU
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randn(num_samples, output_dim)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class StressModel(nn.Module):
    """Large model for GPU stress testing."""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class SystemMonitor:
    """Monitor system resources during tests."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics: List[Dict] = []
        self.thread: Optional[threading.Thread] = None
        self.gpu_available = torch.cuda.is_available() and nvmlInit is not None
        self.gpu_count = 0
        if self.gpu_available:
            try:
                nvmlInit()
                self.gpu_count = nvmlDeviceGetCount()
            except Exception as e:
                print(f"  NVML init failed: {e}")
                self.gpu_available = False
    
    def start(self, interval: float = 1.0):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        while self.monitoring:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None, percpu=True),
                'cpu_avg': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            }
            
            if self.gpu_available:
                gpu_metrics = []
                for i in range(self.gpu_count):
                    try:
                        handle = nvmlDeviceGetHandleByIndex(i)
                        util = nvmlDeviceGetUtilizationRates(handle)
                        mem = nvmlDeviceGetMemoryInfo(handle)
                        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                        
                        gpu_metrics.append({
                            'gpu_id': i,
                            'gpu_util': util.gpu,
                            'memory_util': util.memory,
                            'memory_used_gb': mem.used / (1024**3),
                            'memory_total_gb': mem.total / (1024**3),
                            'temperature_c': temp,
                            'power_w': power
                        })
                    except:
                        pass
                metric['gpu'] = gpu_metrics
            
            self.metrics.append(metric)
            time.sleep(interval)
    
    def get_summary(self) -> Dict:
        """Get summary statistics from collected metrics."""
        if not self.metrics:
            return {}
        
        cpu_avgs = [m['cpu_avg'] for m in self.metrics]
        memory_percents = [m['memory_percent'] for m in self.metrics]
        
        summary = {
            'cpu_avg_max': max(cpu_avgs),
            'cpu_avg_mean': sum(cpu_avgs) / len(cpu_avgs),
            'memory_max_percent': max(memory_percents),
            'memory_mean_percent': sum(memory_percents) / len(memory_percents),
        }
        
        if self.gpu_available and self.metrics[0].get('gpu'):
            gpu_temps = []
            gpu_utils = []
            for m in self.metrics:
                for g in m.get('gpu', []):
                    gpu_temps.append(g['temperature_c'])
                    gpu_utils.append(g['gpu_util'])
            
            if gpu_temps:
                summary['gpu_max_temp'] = max(gpu_temps)
                summary['gpu_avg_temp'] = sum(gpu_temps) / len(gpu_temps)
                summary['gpu_max_util'] = max(gpu_utils)
                summary['gpu_avg_util'] = sum(gpu_utils) / len(gpu_utils)
        
        return summary


class PCVerifier:
    """Main verification suite for PC testing."""
    
    def __init__(self, duration: int = 60, batch_sizes: List[int] = None):
        self.duration = duration
        self.batch_sizes = batch_sizes or [32, 64, 128, 256]
        self.results: List[TestResult] = []
        self.monitor = SystemMonitor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"{Fore.CYAN}PC Verification Suite - AI Research Edition{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=========================================={Style.RESET_ALL}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        print(f"CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print("-" * 50)
    
    def run_all_tests(self) -> List[TestResult]:
        """Execute all verification tests."""
        tests = [
            ("System Info", self.test_system_info),
            ("CPU Stress", self.test_cpu_stress),
            ("Memory Stress", self.test_memory_stress),
            ("Disk I/O", self.test_disk_io),
            ("GPU Memory Bandwidth", self.test_gpu_memory_bandwidth),
            ("GPU Compute Stress", self.test_gpu_compute_stress),
            ("ML Training Simulation", self.test_ml_training),
            ("Mixed Workload", self.test_mixed_workload),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{Fore.YELLOW}Running: {test_name}...{Style.RESET_ALL}")
            try:
                result = test_func()
                self.results.append(result)
                status_color = Fore.GREEN if result.status == "PASS" else Fore.RED
                print(f"{status_color}Result: {result.status}{Style.RESET_ALL}")
                if result.details:
                    print(f"  {result.details}")
            except Exception as e:
                print(f"{Fore.RED}ERROR: {e}{Style.RESET_ALL}")
                self.results.append(TestResult(
                    test_name=test_name,
                    status="ERROR",
                    duration=0,
                    error=str(e)
                ))
        
        return self.results
    
    def test_system_info(self) -> TestResult:
        """Collect and display system information."""
        start = time.time()
        
        info = {
            'cpu_physical': psutil.cpu_count(logical=False),
            'cpu_logical': psutil.cpu_count(logical=True),
            'cpu_freq_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            'ram_gb': psutil.virtual_memory().total / 1024**3,
            'platform': sys.platform,
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = props.total_memory / 1024**3
            info['gpu_multiprocessors'] = props.multi_processor_count
            info['cuda_version'] = torch.version.cuda
            info['pytorch_cuda'] = torch.backends.cudnn.version()
        
        details = "System Information:\n"
        for key, value in info.items():
            details += f"  {key}: {value}\n"
        
        return TestResult(
            test_name="System Info",
            status="PASS",
            duration=time.time() - start,
            details=details,
            metrics=info
        )
    
    def test_cpu_stress(self) -> TestResult:
        """Stress test CPU with intensive computation."""
        start = time.time()
        self.monitor.start(interval=0.5)
        
        # CPU-intensive matrix operations
        def cpu_worker(duration_sec):
            end_time = time.time() + duration_sec
            iterations = 0
            while time.time() < end_time:
                # Heavy matrix multiplication on CPU
                a = torch.randn(1000, 1000)
                b = torch.randn(1000, 1000)
                c = torch.matmul(a, b)
                _ = torch.linalg.eigvals(c[:500, :500])
                iterations += 1
            return iterations
        
        # Run parallel workers on all cores
        threads = []
        num_workers = psutil.cpu_count(logical=True)
        results = [None] * num_workers
        
        def worker_thread(idx, duration):
            results[idx] = cpu_worker(duration)
        
        for i in range(num_workers):
            t = threading.Thread(target=worker_thread, args=(i, self.duration))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.monitor.stop()
        elapsed = time.time() - start
        
        total_iterations = sum(r for r in results if r is not None)
        summary = self.monitor.get_summary()
        
        details = f"CPU Stress completed in {elapsed:.1f}s\n"
        details += f"  Total iterations: {total_iterations:,}\n"
        details += f"  CPU avg utilization: {summary.get('cpu_avg_mean', 0):.1f}%\n"
        details += f"  CPU max utilization: {summary.get('cpu_avg_max', 0):.1f}%"
        
        return TestResult(
            test_name="CPU Stress",
            status="PASS",
            duration=elapsed,
            score=total_iterations / elapsed,
            details=details,
            metrics=summary
        )
    
    def test_memory_stress(self) -> TestResult:
        """Test memory allocation and stress."""
        start = time.time()
        self.monitor.start(interval=0.5)
        
        # Get available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1024**3
        target_gb = min(available_gb * 0.8, mem.total / 1024**3 * 0.75)
        
        allocations = []
        chunk_size = int(target_gb / 10 * 1024**3)  # 10 chunks
        
        try:
            for i in range(10):
                # Allocate and touch memory
                arr = bytearray(chunk_size)
                # Write to ensure pages are committed
                for j in range(0, len(arr), 4096):
                    arr[j] = (i + j) % 256
                allocations.append(arr)
                
                # Small delay to allow monitoring
                time.sleep(0.1)
        except MemoryError:
            pass
        
        # Force cleanup
        allocations.clear()
        import gc
        gc.collect()
        
        self.monitor.stop()
        elapsed = time.time() - start
        summary = self.monitor.get_summary()
        
        details = f"Memory stress completed\n"
        details += f"  Target allocation: {target_gb:.1f} GB\n"
        details += f"  Memory max usage: {summary.get('memory_max_percent', 0):.1f}%\n"
        details += f"  Memory avg usage: {summary.get('memory_mean_percent', 0):.1f}%"
        
        return TestResult(
            test_name="Memory Stress",
            status="PASS",
            duration=elapsed,
            metrics=summary,
            details=details
        )
    
    def test_disk_io(self) -> TestResult:
        """Test disk read/write performance."""
        start = time.time()
        
        test_file = Path("disk_test_temp.bin")
        file_size_mb = 1024  # 1 GB
        chunk_size = 1024 * 1024  # 1 MB chunks
        
        # Write test
        write_start = time.time()
        with open(test_file, 'wb') as f:
            for _ in tqdm(range(file_size_mb), desc="Writing", leave=False):
                f.write(os.urandom(chunk_size))
        write_time = time.time() - write_start
        
        # Read test
        read_start = time.time()
        total_read = 0
        with open(test_file, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
        read_time = time.time() - read_start
        
        # Cleanup
        test_file.unlink()
        
        elapsed = time.time() - start
        write_speed = (file_size_mb / write_time) if write_time > 0 else 0
        read_speed = (file_size_mb / read_time) if read_time > 0 else 0
        
        details = f"Disk I/O completed\n"
        details += f"  Write speed: {write_speed:.1f} MB/s\n"
        details += f"  Read speed: {read_speed:.1f} MB/s\n"
        details += f"  Test file size: {file_size_mb} MB"
        
        metrics = {
            'write_speed_mbps': write_speed,
            'read_speed_mbps': read_speed,
            'write_time': write_time,
            'read_time': read_time
        }
        
        return TestResult(
            test_name="Disk I/O",
            status="PASS",
            duration=elapsed,
            score=(write_speed + read_speed) / 2,
            details=details,
            metrics=metrics
        )
    
    def test_gpu_memory_bandwidth(self) -> TestResult:
        """Test GPU memory bandwidth with large tensor operations."""
        if not torch.cuda.is_available():
            return TestResult(
                test_name="GPU Memory Bandwidth",
                status="SKIP",
                duration=0,
                details="No CUDA GPU available"
            )
        
        start = time.time()
        self.monitor.start(interval=0.5)
        
        device = torch.device('cuda:0')
        
        # Allocate large tensors to test memory bandwidth
        size_gb = 2
        num_elements = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes
        
        # Test copy operations
        copy_times = []
        for _ in range(10):
            a = torch.randn(num_elements, device='cpu')
            torch.cuda.synchronize()
            
            t0 = time.time()
            b = a.to(device)
            torch.cuda.synchronize()
            copy_times.append(time.time() - t0)
            
            del a, b
            torch.cuda.empty_cache()
        
        # Test GPU-GPU copy
        a_gpu = torch.randn(num_elements, device=device)
        torch.cuda.synchronize()
        
        gpu_copy_times = []
        for _ in range(10):
            t0 = time.time()
            b_gpu = a_gpu.clone()
            torch.cuda.synchronize()
            gpu_copy_times.append(time.time() - t0)
        
        del a_gpu
        torch.cuda.empty_cache()
        
        self.monitor.stop()
        elapsed = time.time() - start
        summary = self.monitor.get_summary()
        
        avg_copy_time = sum(copy_times) / len(copy_times)
        avg_gpu_copy_time = sum(gpu_copy_times) / len(gpu_copy_times)
        bandwidth_gbps = (size_gb * 2) / avg_copy_time  # H2D + D2H
        
        details = f"GPU Memory Bandwidth test completed\n"
        details += f"  H2D copy time: {avg_copy_time*1000:.1f} ms\n"
        details += f"  GPU-GPU copy time: {avg_gpu_copy_time*1000:.1f} ms\n"
        details += f"  Estimated bandwidth: {bandwidth_gbps:.1f} GB/s\n"
        details += f"  GPU max temp: {summary.get('gpu_max_temp', 'N/A')}"
        
        return TestResult(
            test_name="GPU Memory Bandwidth",
            status="PASS",
            duration=elapsed,
            score=bandwidth_gbps,
            details=details,
            metrics={'bandwidth_gbps': bandwidth_gbps, 'temp': summary.get('gpu_max_temp')}
        )
    
    def test_gpu_compute_stress(self) -> TestResult:
        """Heavy GPU compute stress test using matrix operations."""
        if not torch.cuda.is_available():
            return TestResult(
                test_name="GPU Compute Stress",
                status="SKIP",
                duration=0,
                details="No CUDA GPU available"
            )
        
        start = time.time()
        self.monitor.start(interval=0.5)
        
        device = torch.device('cuda:0')
        
        # Large matrix for intensive compute
        size = 8192
        iterations = 0
        end_time = time.time() + self.duration
        
        # Pre-allocate tensors
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"  Running GPU compute stress on {size}x{size} matrices...")
        
        with tqdm(total=self.duration, desc="GPU Stress", unit="s") as pbar:
            last_update = time.time()
            while time.time() < end_time:
                # Heavy compute operations
                c = torch.matmul(a, b)
                d = torch.relu(c)
                e = torch.softmax(d, dim=1)
                
                # FFT operations
                f = torch.fft.fft2(a[:4096, :4096])
                _ = torch.abs(f)
                
                # Eigenvalue computation
                _ = torch.linalg.eigvals(a[:1024, :1024])
                
                iterations += 1
                torch.cuda.synchronize()
                
                # Update progress bar
                if time.time() - last_update >= 1:
                    pbar.update(int(time.time() - last_update))
                    last_update = time.time()
        
        del a, b, c, d, e, f
        torch.cuda.empty_cache()
        
        self.monitor.stop()
        elapsed = time.time() - start
        summary = self.monitor.get_summary()
        
        ops_per_iter = size**3 + size**2 * 3  # Rough estimate
        gflops = (iterations * ops_per_iter) / (elapsed * 1e9)
        
        details = f"GPU Compute stress completed\n"
        details += f"  Iterations: {iterations:,}\n"
        details += f"  Performance: {gflops:.1f} GFLOPS\n"
        details += f"  GPU max temp: {summary.get('gpu_max_temp', 'N/A')}°C\n"
        details += f"  GPU avg util: {summary.get('gpu_avg_util', 0):.1f}%\n"
        details += f"  GPU max util: {summary.get('gpu_max_util', 0)}%"
        
        return TestResult(
            test_name="GPU Compute Stress",
            status="PASS",
            duration=elapsed,
            score=gflops,
            details=details,
            metrics=summary
        )
    
    def test_ml_training(self) -> TestResult:
        """Simulate ML training with synthetic data."""
        if not torch.cuda.is_available():
            return TestResult(
                test_name="ML Training Simulation",
                status="SKIP",
                duration=0,
                details="No CUDA GPU available - would train on CPU (slow)"
            )
        
        start = time.time()
        self.monitor.start(interval=1.0)
        
        device = torch.device('cuda:0')
        
        # Create model
        model = StressModel(input_dim=1024, hidden_dim=4096, output_dim=256).to(device)
        
        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scaler = GradScaler('cuda')  # Mixed precision (PyTorch 2.6+)
        
        # Synthetic dataset
        dataset = SyntheticDataset(num_samples=50000, input_dim=1024, output_dim=256)
        
        results_by_batch = []
        
        for batch_size in self.batch_sizes:
            try:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                
                model.train()
                samples_processed = 0
                epoch_start = time.time()
                losses = []
                
                # Run for ~10 seconds per batch size
                run_until = time.time() + max(10, self.duration // len(self.batch_sizes))
                batches = 0
                
                for data, target in dataloader:
                    if time.time() > run_until:
                        break
                    
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Mixed precision forward
                    with autocast('cuda'):  # PyTorch 2.6+ API
                        output = model(data)
                        loss = criterion(output, target)
                    
                    # Backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    losses.append(loss.item())
                    samples_processed += len(data)
                    batches += 1
                    
                    torch.cuda.synchronize()
                
                elapsed = time.time() - epoch_start
                throughput = samples_processed / elapsed if elapsed > 0 else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                
                results_by_batch.append({
                    'batch_size': batch_size,
                    'throughput': throughput,
                    'avg_loss': avg_loss,
                    'batches': batches,
                    'time': elapsed
                })
                
                print(f"  Batch {batch_size}: {throughput:.0f} samples/sec, loss={avg_loss:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Batch {batch_size}: OOM - stopping batch size tests")
                    torch.cuda.empty_cache()
                    break
                raise
        
        del model, optimizer, dataset
        torch.cuda.empty_cache()
        
        self.monitor.stop()
        elapsed = time.time() - start
        summary = self.monitor.get_summary()
        
        # Find best throughput
        best = max(results_by_batch, key=lambda x: x['throughput']) if results_by_batch else None
        
        details = f"ML Training simulation completed\n"
        if best:
            details += f"  Best batch size: {best['batch_size']}\n"
            details += f"  Max throughput: {best['throughput']:.0f} samples/sec\n"
        details += f"  GPU max temp: {summary.get('gpu_max_temp', 'N/A')}°C\n"
        details += f"  Tested batch sizes: {[r['batch_size'] for r in results_by_batch]}"
        
        return TestResult(
            test_name="ML Training Simulation",
            status="PASS",
            duration=elapsed,
            score=best['throughput'] if best else 0,
            details=details,
            metrics={
                'batch_results': results_by_batch,
                'monitoring': summary
            }
        )
    
    def test_mixed_workload(self) -> TestResult:
        """Combined CPU+GPU+Memory stress test."""
        start = time.time()
        self.monitor.start(interval=0.5)
        
        stop_event = threading.Event()
        
        def cpu_worker():
            while not stop_event.is_set():
                a = torch.randn(500, 500)
                b = torch.randn(500, 500)
                _ = torch.matmul(a, b)
        
        def memory_worker():
            buffers = []
            while not stop_event.is_set():
                try:
                    buffers.append(bytearray(50 * 1024 * 1024))  # 50MB
                    if len(buffers) > 20:
                        buffers.pop(0)
                except MemoryError:
                    buffers.clear()
                time.sleep(0.1)
        
        # Start CPU and memory workers
        cpu_threads = [threading.Thread(target=cpu_worker) for _ in range(psutil.cpu_count(logical=True)//2)]
        mem_thread = threading.Thread(target=memory_worker)
        
        for t in cpu_threads:
            t.daemon = True
            t.start()
        mem_thread.daemon = True
        mem_thread.start()
        
        # GPU workload
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = StressModel(input_dim=1024, hidden_dim=2048, output_dim=256).to(device)
            
            data = torch.randn(256, 1024, device=device)
            target = torch.randn(256, 256, device=device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            iterations = 0
            end_time = time.time() + self.duration
            
            print(f"  Running mixed workload (CPU + Memory + GPU)...")
            
            with tqdm(total=self.duration, desc="Mixed", unit="s") as pbar:
                last_update = time.time()
                while time.time() < end_time:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    iterations += 1
                    
                    if time.time() - last_update >= 1:
                        pbar.update(int(time.time() - last_update))
                        last_update = time.time()
            
            del model, data, target, optimizer
            torch.cuda.empty_cache()
        
        stop_event.set()
        for t in cpu_threads:
            t.join(timeout=1)
        mem_thread.join(timeout=1)
        
        self.monitor.stop()
        elapsed = time.time() - start
        summary = self.monitor.get_summary()
        
        details = f"Mixed workload test completed\n"
        details += f"  Duration: {elapsed:.1f}s\n"
        details += f"  CPU max: {summary.get('cpu_avg_max', 0):.1f}%\n"
        details += f"  Memory max: {summary.get('memory_max_percent', 0):.1f}%\n"
        if 'gpu_max_temp' in summary:
            details += f"  GPU max temp: {summary['gpu_max_temp']}°C\n"
            details += f"  GPU max util: {summary['gpu_max_util']}%"
        
        return TestResult(
            test_name="Mixed Workload",
            status="PASS",
            duration=elapsed,
            score=iterations / elapsed if 'iterations' in dir() else 0,
            details=details,
            metrics=summary
        )
    
    def generate_report(self, output_file: str = "verification_report.json"):
        """Generate JSON and console report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_count': psutil.cpu_count(),
                'ram_gb': psutil.virtual_memory().total / 1024**3,
                'gpu_available': torch.cuda.is_available(),
            },
            'results': [r.to_dict() for r in self.results],
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results if r.status == "PASS"),
                'skipped': sum(1 for r in self.results if r.status == "SKIP"),
                'errors': sum(1 for r in self.results if r.status == "ERROR"),
                'total_duration': sum(r.duration for r in self.results)
            }
        }
        
        if torch.cuda.is_available():
            report['system']['gpu_name'] = torch.cuda.get_device_name(0)
            report['system']['gpu_count'] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            report['system']['gpu_memory_gb'] = props.total_memory / 1024**3
        
        # Save JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{Fore.CYAN}Report saved to: {output_file}{Style.RESET_ALL}")
        
        # Console summary table
        print(f"\n{Fore.CYAN}================== SUMMARY =================={Style.RESET_ALL}")
        table_data = []
        for r in self.results:
            status_color = Fore.GREEN if r.status == "PASS" else (Fore.YELLOW if r.status == "SKIP" else Fore.RED)
            score_str = f"{r.score:.1f}" if r.score else "-"
            table_data.append([
                r.test_name,
                f"{status_color}{r.status}{Style.RESET_ALL}",
                f"{r.duration:.1f}s",
                score_str
            ])
        
        headers = ["Test", "Status", "Duration", "Score"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
        print(f"  Total tests: {report['summary']['total_tests']}")
        print(f"  Passed: {Fore.GREEN}{report['summary']['passed']}{Style.RESET_ALL}")
        print(f"  Skipped: {Fore.YELLOW}{report['summary']['skipped']}{Style.RESET_ALL}")
        print(f"  Errors: {Fore.RED}{report['summary']['errors']}{Style.RESET_ALL}")
        print(f"  Total duration: {report['summary']['total_duration']:.1f}s")
        
        # Overall verdict
        if report['summary']['errors'] == 0 and report['summary']['passed'] >= report['summary']['total_tests'] // 2:
            print(f"\n{Fore.GREEN}✓ PC VERIFICATION PASSED{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ PC VERIFICATION ISSUES DETECTED{Style.RESET_ALL}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="PC Verification Suite for AI Research")
    parser.add_argument("--duration", type=int, default=60, help="Duration per stress test in seconds")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Batch sizes to test for ML training")
    parser.add_argument("--output", type=str, default="verification_report.json",
                        help="Output JSON report file")
    args = parser.parse_args()
    
    print(f"\n{Fore.CYAN}Starting PC Verification Suite...{Style.RESET_ALL}")
    print(f"Stress test duration: {args.duration} seconds per test")
    print(f"ML batch sizes: {args.batch_sizes}")
    print()
    
    verifier = PCVerifier(duration=args.duration, batch_sizes=args.batch_sizes)
    verifier.run_all_tests()
    verifier.generate_report(output_file=args.output)


if __name__ == "__main__":
    main()
