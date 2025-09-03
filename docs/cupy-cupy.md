<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: Accelerate Your Python with GPU-Powered NumPy & SciPy

**CuPy is a powerful library that enables you to run your familiar NumPy and SciPy code on NVIDIA GPUs or AMD ROCm platforms, significantly boosting performance.**

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

[**Website**](https://cupy.dev/) | [**Install**](https://docs.cupy.dev/en/stable/install.html) | [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html) | [**Examples**](https://github.com/cupy/cupy/tree/main/examples) | [**Documentation**](https://docs.cupy.dev/en/stable/) | [**API Reference**](https://docs.cupy.dev/en/stable/reference/) | [**Forum**](https://groups.google.com/forum/#!forum/cupy)

## Key Features

*   **NumPy & SciPy Compatibility:** CuPy offers a drop-in replacement for NumPy and SciPy, allowing you to leverage existing code with minimal changes.
*   **GPU Acceleration:** Execute your numerical computations on NVIDIA GPUs (CUDA) or AMD GPUs (ROCm) for significant performance gains.
*   **CUDA Integration:** Access low-level CUDA features, including RawKernels, Streams, and CUDA Runtime APIs, for advanced customization.
*   **Easy to Use:** Simply replace `numpy` with `cupy` in your code to get started.
*   **Cross-Platform Support:** Works on Linux and Windows, and provides pre-built packages for different CUDA and ROCm versions.

## Getting Started

CuPy simplifies GPU-accelerated computing with Python, providing a familiar NumPy/SciPy-compatible array library.

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
# Output:
# array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype=float32)
print(x.sum(axis=1))
# Output:
# array([  3.,  12.], dtype=float32)
```

## Installation

### Pip

Install pre-built binary packages (wheels) for your platform via pip.

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

Install via conda-forge.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation: `conda install -c conda-forge cupy-core`.

For specific CUDA versions: `conda install -c conda-forge cupy cuda-version=12.0`.

> [!NOTE]\
> If you encounter problems with CuPy installed from `conda-forge`, report to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy container images using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   [**Installation Guide**](https://docs.cupy.dev/en/stable/install.html) - detailed installation instructions
*   [**Release Notes**](https://github.com/cupy/cupy/releases) - view the latest updates and changes
*   [**Projects using CuPy**](https://github.com/cupy/cupy/wiki/Projects-using-CuPy) - explore projects leveraging CuPy
*   [**Contribution Guide**](https://docs.cupy.dev/en/stable/contribution.html) - learn how to contribute
*   [GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/) - discover how to accelerate using CuPy
*   [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial) - learn about signal processing

[^1]: cuSignal is now part of CuPy starting v13.0.0.

## License

CuPy is licensed under the [MIT License](https://github.com/cupy/cupy/blob/main/LICENSE).

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

[**Back to Top**](https://github.com/cupy/cupy)