html
<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy: Accelerate Your Python Code with GPU Power

CuPy is a powerful, NumPy and SciPy-compatible array library that leverages the parallel processing capabilities of GPUs to significantly speed up your scientific computing tasks.  (<a href="https://github.com/cupy/cupy">See the original repository</a>)

[![pypi](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

*   **Website:** [https://cupy.dev/](https://cupy.dev/)
*   **Install:** [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html)
*   **Tutorial:** [https://docs.cupy.dev/en/stable/user_guide/basic.html](https://docs.cupy.dev/en/stable/user_guide/basic.html)
*   **Examples:** [https://github.com/cupy/cupy/tree/main/examples](https://github.com/cupy/cupy/tree/main/examples)
*   **Documentation:** [https://docs.cupy.dev/en/stable/](https://docs.cupy.dev/en/stable/)
*   **API Reference:** [https://docs.cupy.dev/en/stable/reference/](https://docs.cupy.dev/en/stable/reference/)
*   **Forum:** [https://groups.google.com/forum/#!forum/cupy](https://groups.google.com/forum/#!forum/cupy)

## Key Features

*   **NumPy & SciPy Compatibility:** CuPy provides a familiar API, making it easy to migrate existing NumPy and SciPy code.
*   **GPU Acceleration:**  Executes array operations on NVIDIA CUDA and AMD ROCm platforms, providing significant performance gains.
*   **Drop-in Replacement:** Easily accelerate your code by replacing `numpy` with `cupy` with minimal code changes.
*   **CUDA Integration:** Offers access to low-level CUDA features, including RawKernels, Streams, and CUDA Runtime APIs for advanced users.

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

Install pre-built binary packages for common platforms.

**Note:** Choose the correct package that matches your CUDA version.

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

Install CuPy using Conda-Forge.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation without CUDA dependencies: `conda install -c conda-forge cupy-core`.

For specific CUDA versions: `conda install -c conda-forge cupy cuda-version=12.0`.

> [!NOTE]\
> Report any issues with Conda-Forge installations to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Use the NVIDIA Container Toolkit.

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   **Installation Guide:** [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html) - Build from source instructions.
*   **Release Notes:** [https://github.com/cupy/cupy/releases](https://github.com/cupy/cupy/releases)
*   **Projects using CuPy:** [https://github.com/cupy/cupy/wiki/Projects-using-CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   **Contribution Guide:** [https://docs.cupy.dev/en/stable/contribution.html](https://docs.cupy.dev/en/stable/contribution.html)
*   **GTC Session:** [GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   **ICASSP Tutorial:** [GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)](https://github.com/awthomp/cusignal-icassp-tutorial)

[^1]: cuSignal is now part of CuPy starting v13.0.0.

## License

MIT License (see `LICENSE` file).

CuPy is based on NumPy and SciPy (see `docs/source/license.rst` file).

Developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

## Reference

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```