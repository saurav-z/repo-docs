<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: Accelerate Your Python Code with GPU Power

**CuPy empowers you to run your NumPy and SciPy code on NVIDIA CUDA or AMD ROCm GPUs, significantly speeding up your computations.**  [Visit the original repository](https://github.com/cupy/cupy).

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

## Key Features

*   **NumPy & SciPy Compatibility:**  CuPy offers a seamless drop-in replacement for NumPy and SciPy, allowing you to leverage GPU acceleration with minimal code changes.
*   **GPU Acceleration:**  Execute your array-based computations on NVIDIA GPUs (CUDA) or AMD GPUs (ROCm) for significant performance gains.
*   **Low-Level CUDA Access:**  Provides access to low-level CUDA features for advanced users, including raw kernels, streams, and CUDA runtime APIs.
*   **Easy Integration:** Works with existing CUDA C/C++ programs via RawKernels.

## Installation

### Pip

Install CuPy using pip. Binary wheels are available for various CUDA versions, ROCm and platforms.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

> [!NOTE]\
> To install pre-releases, append `--pre -U -f https://pip.cupy.dev/pre` (e.g., `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre`).

### Conda

Install CuPy using conda-forge.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

If you need a slim installation (without also getting CUDA dependencies installed), you can do `conda install -c conda-forge cupy-core`.

If you need to use a particular CUDA version (say 12.0), you can use the `cuda-version` metapackage to select the version, e.g. `conda install -c conda-forge cupy cuda-version=12.0`.

> [!NOTE]\
> If you encounter any problem with CuPy installed from `conda-forge`, please feel free to report to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues), and we will help investigate if it is just a packaging issue in `conda-forge`'s recipe or a real issue in CuPy.

### Docker

Use [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to run [CuPy container images](https://hub.docker.com/r/cupy/cupy).

```bash
docker run --gpus all -it cupy/cupy
```

## Quick Start

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
print(x.sum(axis=1))
```

## Resources

*   [Website](https://cupy.dev/)
*   [Installation Guide](https://docs.cupy.dev/en/stable/install.html)
*   [Tutorial](https://docs.cupy.dev/en/stable/user_guide/basic.html)
*   [Examples](https://github.com/cupy/cupy/tree/main/examples)
*   [Documentation](https://docs.cupy.dev/en/stable/)
*   [API Reference](https://docs.cupy.dev/en/stable/reference/)
*   [Forum](https://groups.google.com/forum/#!forum/cupy)
*   [Release Notes](https://github.com/cupy/cupy/releases)
*   [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)
*   [GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial)

[^1]: cuSignal is now part of CuPy starting v13.0.0.

## License

MIT License (see `LICENSE` file).

CuPy is designed based on NumPy's API and SciPy's API (see `docs/source/license.rst` file).

CuPy is being developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

## Reference

Ryosuke Okuta, Yuya Unno, Daisuke Nishino, Shohei Hido and Crissman Loomis.
**CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations.**
*Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017).
[[PDF](http://learningsys.org/nips17/assets/papers/paper_16.pdf)]

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```