<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo"/>
</div>

# CuPy: GPU-Accelerated NumPy & SciPy for Python

**Supercharge your Python numerical computations with CuPy, a drop-in replacement for NumPy and SciPy, optimized for NVIDIA CUDA and AMD ROCm GPUs.**  This allows you to leverage the power of your GPU for faster calculations.

[View the CuPy Repository on GitHub](https://github.com/cupy/cupy)

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub License](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

**Key Features:**

*   **NumPy & SciPy Compatibility:** CuPy offers a familiar API, making it easy to migrate existing NumPy and SciPy code.
*   **GPU Acceleration:**  Execute your numerical computations on NVIDIA CUDA and AMD ROCm GPUs for significant performance gains.
*   **Drop-in Replacement:**  CuPy can often be used as a direct replacement for NumPy/SciPy, requiring minimal code changes.
*   **Low-Level CUDA Access:** Access advanced CUDA features like raw kernels, streams, and CUDA Runtime APIs for fine-grained control and optimization.
*   **Comprehensive Documentation:** Extensive documentation, tutorials, and examples to help you get started quickly.

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

Choose your preferred installation method:

### Pip

Binary packages are available on [PyPI](https://pypi.org/org/cupy/) for Linux and Windows.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86\_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86\_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86\_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86\_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86\_64            | `pip install cupy-rocm-5-0`                                   |

> [!NOTE]\
> Install pre-releases with `--pre -U -f https://pip.cupy.dev/pre` (e.g., `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre`).

### Conda

Binary packages are available on [Conda-Forge](https://anaconda.org/conda-forge/cupy) for Linux and Windows.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86\_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation: `conda install -c conda-forge cupy-core`.  To specify a CUDA version: `conda install -c conda-forge cupy cuda-version=12.0`.

> [!NOTE]\
> Report issues with Conda-Forge installations to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Use the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to run CuPy container images from [Docker Hub](https://hub.docker.com/r/cupy/cupy).

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   [Installation Guide](https://docs.cupy.dev/en/stable/install.html)
*   [Release Notes](https://github.com/cupy/cupy/releases)
*   [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)
*   [GPU Acceleration in Python using CuPy and Numba (GTC November 2021)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial)

## License

MIT License (see `LICENSE` file). CuPy is based on NumPy and SciPy (see `docs/source/license.rst`).

## Acknowledgements

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