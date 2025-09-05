<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: GPU-Accelerated NumPy & SciPy for Lightning-Fast Python

**Supercharge your Python numerical computations with CuPy, a powerful library that brings the familiar NumPy and SciPy experience to NVIDIA GPUs and AMD ROCm platforms.** ([Original Repo](https://github.com/cupy/cupy))

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

**Key Features:**

*   **NumPy & SciPy Compatibility:** CuPy acts as a drop-in replacement for NumPy/SciPy, making it easy to accelerate existing code with minimal changes.
*   **GPU Acceleration:** Execute your numerical computations on NVIDIA CUDA or AMD ROCm-enabled GPUs for significant performance gains.
*   **CUDA Integration:**  Directly access low-level CUDA features like RawKernels, Streams, and CUDA Runtime APIs for advanced control and optimization.
*   **Easy to Use:** Benefit from a familiar API that mirrors NumPy/SciPy, lowering the barrier to entry for GPU programming.
*   **Broad Support:**  Supports a wide range of hardware and software configurations, including CUDA and ROCm.

**Quick Example:**

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
# array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype=float32)
print(x.sum(axis=1))
# array([  3.,  12.], dtype=float32)
```

**Installation:**

Choose your preferred method to install CuPy.

### Pip

Install pre-built binary packages for Linux and Windows from PyPI.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

> **Note:**  Install pre-releases with `--pre -U -f https://pip.cupy.dev/pre`.

### Conda

Install binary packages for Linux and Windows via Conda-Forge.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

*   For a slim install (without CUDA dependencies):  `conda install -c conda-forge cupy-core`.
*   To specify a CUDA version (e.g., 12.0): `conda install -c conda-forge cupy cuda-version=12.0`.

> **Note:**  Report Conda-Forge issues to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy container images using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

```bash
docker run --gpus all -it cupy/cupy
```

**Resources:**

*   [Installation Guide](https://docs.cupy.dev/en/stable/install.html)
*   [Release Notes](https://github.com/cupy/cupy/releases)
*   [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)
*   [GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial)

[^1]: cuSignal is now part of CuPy starting v13.0.0.

**License:**

MIT License (see `LICENSE` file).  CuPy is built upon NumPy/SciPy APIs (see `docs/source/license.rst`).

**Developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).**

**Reference:**

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}