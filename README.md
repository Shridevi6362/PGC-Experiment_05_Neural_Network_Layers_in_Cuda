# Experiment 5: Neural Network Layers in CUDA

## Overview

This project implements and evaluates fundamental neural-network forward-pass layers using custom CUDA kernels in C++. It executes a two-layer feed-forward inference pipeline on both the CPU and GPU, validates numerical equivalence, measures execution latencies, and calculates GPU speedup.

### Network Architecture

The forward pass implements inference (no training or backpropagation) for a two-layer dense network:

1. **Hidden Layer:** $Z_1 = X \times W_1 + b_1$, followed by $A_1 = \text{ReLU}(Z_1)$
2. **Output Layer:** $Z_2 = A_1 \times W_2 + b_2$, followed by $Y = \text{Sigmoid}(Z_2)$

| Parameter / Tensor | Shape / Value |
| :--- | :--- |
| **Batch Size (`BATCH`)** | $256$ |
| **Input (`X`)** | $256 \times 128$ |
| **Weights 1 (`W1`)** | $128 \times 64$ |
| **Bias 1 (`b1`)** | $64$ |
| **Hidden Activation (`A1`)** | $256 \times 64$ |
| **Weights 2 (`W2`)** | $64 \times 10$ |
| **Bias 2 (`b2`)** | $10$ |
| **Final Output (`Y`)** | $256 \times 10$ |
| **Benchmark Repeats** | $20$ |

## Implemented CUDA Kernels

* **`matrixMulBiasKernel`**: Computes $C[\text{row}, \text{col}] = \sum_{k} A[\text{row}, k] \cdot B[k, \text{col}] + \text{bias}[\text{col}]$. Uses a 2D grid of $16 \times 16$ thread blocks where each thread computes one output element.
* **`reluKernel`**: Applies $\text{ReLU}(x) = \max(0, x)$ element-wise across a 1D grid ($256$ threads per block).
* **`sigmoidKernel`**: Applies $\sigma(x) = \frac{1}{1 + e^{-x}}$ element-wise across a 1D grid ($256$ threads per block).

## Prerequisites

* **Hardware:** NVIDIA CUDA-capable GPU
* **OS:** Windows 10/11 (64-bit)
* **Software:**
  * NVIDIA GPU Driver
  * NVIDIA CUDA Toolkit (`nvcc`)
  * Visual Studio 2022 Build Tools with **Desktop development with C++** (`MSVC v143` & Windows SDK)

## Build & Execution Instructions (Windows)

1. **Open the Developer Command Prompt:**
   Search the Windows Start menu for **x64 Native Tools Command Prompt for VS 2022** (or *Developer Command Prompt for VS 2022*). *Do not use standard PowerShell, as `nvcc` requires `cl.exe` in its environment PATH.*

2. **Verify Environment:**
   ```cmd
   nvidia-smi
   nvcc --version
   where cl
   ```

3. **Navigate to Your Project Directory:**
   Ensure you are in a user-writable folder (avoid `C:\Windows\System32`):
   ```cmd
   mkdir "%USERPROFILE%\Documents\Experiment5_CUDA"
   cd /d "%USERPROFILE%\Documents\Experiment5_CUDA"
   ```

4. **Compile the Source File:**
   Place `neural_network.cu` in the directory and compile with optimization flag `-O2`:
   ```cmd
   nvcc -O2 neural_network.cu -o neural_network.exe
   ```

5. **Run the Executable:**
   ```cmd
   neural_network.exe
   ```

## Timing & Validation Methodology

* **Unit Tests:** Before benchmarking, `testMatrixKernel`, `testReLUKernel`, and `testSigmoidKernel` verify each CUDA kernel against known analytical values.
* **Warm-up:** Runs one un-timed forward pass on the GPU to eliminate cold-start/context initialization overhead.
* **Kernel-Only Timing:** Measured using `cudaEventRecord` with data already resident in GPU global memory.
* **End-to-End Timing:** Measured using `std::chrono::steady_clock`, capturing host-to-device memory transfers (`cudaMemcpyHostToDevice`), kernel execution, device synchronization, and device-to-host output retrieval (`cudaMemcpyDeviceToHost`).
* **Numerical Tolerance:** Passes validation if $\max |Y_{\text{CPU}} - Y_{\text{GPU}}| < 10^{-4}$.

## Reference Output (NVIDIA RTX 4500 Ada Generation)

```text
EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA
GPU detected: NVIDIA RTX 4500 Ada Generation

Testing individual CUDA kernels...
Matrix multiplication + bias: PASSED
ReLU: PASSED
Sigmoid: PASSED

Forward-pass dimensions:
  Input:        256 x 128
  Hidden layer: 256 x 64
  Output:       256 x 10

Average CPU compute time:     0.526860 ms
Average GPU kernel-only time: 0.026517 ms
Average GPU end-to-end time:  0.066010 ms
Kernel-only speedup:          19.868913x
End-to-end speedup:           7.981518x

Maximum absolute output difference: 0.000000
CPU and GPU outputs: MATCH (within tolerance)

First 10 output values (GPU):
0.498196 0.503101 0.508006 0.512911 0.517814 0.497729 0.502638 0.507548 0.512457 0.517364
```
## Repository structure

Experiment5_CUDA/
├── screenshot/
│   └── output.png             # Terminal output screenshot from running neural_network.exe
├── README.md                  # Experiment documentation, setup guide, and results
└── neural_network.cu          # Complete CUDA C++ source code
