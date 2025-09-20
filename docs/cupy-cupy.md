<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy: GPU-Accelerated NumPy and SciPy for Python

**Supercharge your Python numerical computations with CuPy, a NumPy-compatible array library that harnesses the power of GPUs for blazing-fast performance.**  

[See the original repo](https://github.com/cupy/cupy)

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

## Key Features

*   **NumPy & SciPy Compatibility:** Seamlessly run your existing NumPy/SciPy code on NVIDIA CUDA or AMD ROCm platforms with minimal code changes.
*   **GPU Acceleration:** Leverage the parallel processing power of GPUs for significant speedups in numerical computations.
*   **Drop-in Replacement:** CuPy acts as a drop-in replacement for NumPy, making it easy to integrate into your existing projects.
*   **CUDA & ROCm Support:** Supports both NVIDIA CUDA and AMD ROCm platforms.
*   **Low-Level CUDA Access:** Access low-level CUDA features, including raw kernels, streams, and runtime APIs, for advanced customization and performance tuning.

## Installation

### Pip

Install pre-built binary packages for various CUDA and ROCm versions using pip:

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*experimental*)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*experimental*)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

> [!NOTE]\
> To install pre-releases, append `--pre -U -f https://pip.cupy.dev/pre` (e.g., `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre`).

### Conda

Install CuPy using Conda-Forge:

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation, use `conda install -c conda-forge cupy-core`.

To specify a CUDA version: `conda install -c conda-forge cupy cuda-version=12.0`.

> [!NOTE]\
> Report any issues with conda-forge packages to the [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy container images using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html):

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   [Installation Guide](https://docs.cupy.dev/en/stable/install.html)
*   [Release Notes](https://github.com/cupy/cupy/releases)
*   [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)
*   [GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial)

[^1]: cuSignal is now part of CuPy starting v13.0.0.

## License

MIT License (see `LICENSE` file).

CuPy is based on NumPy's and SciPy's APIs (see `docs/source/license.rst`).

CuPy is developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

## Reference

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}