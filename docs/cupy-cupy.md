<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy: Accelerate Your Python Code with GPU Power

**CuPy is a powerful, NumPy-compatible library that brings GPU acceleration to your Python scientific computing workflows.** [Learn more at the original repository](https://github.com/cupy/cupy).

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

## Key Features

*   **NumPy/SciPy Compatibility:** Seamlessly run your existing NumPy and SciPy code on NVIDIA CUDA or AMD ROCm GPUs with minimal changes.
*   **Drop-in Replacement:** Utilize CuPy as a direct replacement for NumPy, benefiting from GPU acceleration without significant code modification.
*   **High Performance:** Leverage the power of GPUs for faster computation, especially for array-based operations.
*   **CUDA Integration:** Access low-level CUDA features, including RawKernels, Streams, and CUDA Runtime APIs, for advanced customization and optimization.
*   **Broad Compatibility:** Supports various CUDA and ROCm versions, as well as different architectures.

## Quick Example

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
# array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype=float32)
print(x.sum(axis=1))
# array([  3.,  12.], dtype=float32)
```

## Installation

### Pip

Install CuPy using pip for various CUDA and ROCm versions:

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86\_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86\_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86\_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*experimental*)          | x86\_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*experimental*)          | x86\_64            | `pip install cupy-rocm-5-0`                                   |

> \[!NOTE]\
> Install pre-releases using `--pre -U -f https://pip.cupy.dev/pre`.

### Conda

Install CuPy via Conda-Forge:

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86\_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation (without CUDA dependencies), use `conda install -c conda-forge cupy-core`. Specify CUDA version using `cuda-version=12.0`.

> \[!NOTE]\
> Report issues with Conda-Forge installations to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) and CuPy container images:

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

## Contributors

CuPy is developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

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