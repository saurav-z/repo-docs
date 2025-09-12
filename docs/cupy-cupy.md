<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo"/>
</div>

# CuPy: Accelerate Your Python with GPU-Powered NumPy & SciPy

**CuPy brings the familiar NumPy and SciPy experience to your NVIDIA CUDA and AMD ROCm GPUs, dramatically accelerating your scientific computing workflows.** [Explore the CuPy Repository](https://github.com/cupy/cupy)

[![PyPI](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub License](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

[**Website**](https://cupy.dev/) | [**Install**](https://docs.cupy.dev/en/stable/install.html) | [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html) | [**Examples**](https://github.com/cupy/cupy/tree/main/examples) | [**Documentation**](https://docs.cupy.dev/en/stable/) | [**API Reference**](https://docs.cupy.dev/en/stable/reference/) | [**Forum**](https://groups.google.com/forum/#!forum/cupy)

## Key Features

*   **NumPy & SciPy Compatibility:**  CuPy is designed as a drop-in replacement for NumPy and SciPy, so you can use your existing code with minimal changes.
*   **GPU Acceleration:** Leverage the power of NVIDIA CUDA and AMD ROCm GPUs for significant performance gains in numerical computation.
*   **Seamless Integration:**  Easily move your NumPy/SciPy code to the GPU.
*   **Low-Level CUDA Access:** Provides access to low-level CUDA features like RawKernels, Streams, and CUDA Runtime APIs for advanced control and optimization.

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

Install CuPy using `pip` for various CUDA and ROCm platforms. Binary wheels are available for Linux and Windows on [PyPI](https://pypi.org/org/cupy/).

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*[experimental](https://docs.cupy.dev/en/latest/install.html#using-cupy-on-amd-gpu-experimental)*)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

> **Note:** To install pre-releases, use `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre` (replace `cupy-cuda11x` with the appropriate package).

### Conda

Install CuPy using `conda` for Linux and Windows on [Conda-Forge](https://anaconda.org/conda-forge/cupy).

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For slim installations without CUDA dependencies: `conda install -c conda-forge cupy-core`.

Specify a CUDA version (e.g., 12.0) with: `conda install -c conda-forge cupy cuda-version=12.0`.

> **Note:** Report issues with `conda-forge` installations to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) and [CuPy container images](https://hub.docker.com/r/cupy/cupy).

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

CuPy is based on NumPy's and SciPy's APIs (see `docs/source/license.rst` file).

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