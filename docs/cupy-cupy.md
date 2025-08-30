html
<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: GPU-Accelerated NumPy and SciPy for Python

**Supercharge your Python numerical computing with CuPy, a NumPy/SciPy-compatible library that harnesses the power of NVIDIA GPUs and AMD ROCm.**  <a href="https://github.com/cupy/cupy"> View the original repository.</a>

CuPy seamlessly integrates with your existing NumPy and SciPy code, providing significant performance gains for computationally intensive tasks.

## Key Features

*   **NumPy/SciPy Compatibility:** CuPy acts as a drop-in replacement for NumPy and SciPy, allowing you to run existing code on GPUs with minimal changes.
*   **GPU Acceleration:** Leverage the massive parallel processing capabilities of NVIDIA GPUs and AMD ROCm to accelerate your computations.
*   **CUDA and ROCm Support:** Compatible with NVIDIA CUDA and AMD ROCm platforms, providing flexibility in hardware choice.
*   **Low-Level CUDA Access:** Offers access to low-level CUDA features for advanced users, including RawKernels, Streams, and CUDA Runtime APIs.
*   **Easy Installation:** Simple installation via pip, conda, and Docker.

## Installation

### Pip

Install CuPy using pip, selecting the appropriate package for your CUDA or ROCm environment.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86\_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86\_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86\_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (*experimental*)          | x86\_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (*experimental*)          | x86\_64            | `pip install cupy-rocm-5-0`                                   |

> **Note:** To install pre-releases, use the `--pre -U -f https://pip.cupy.dev/pre` flag (e.g., `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre`).

### Conda

Install CuPy via Conda-Forge for easy environment management.

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86\_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation (without CUDA dependencies), use `conda install -c conda-forge cupy-core`. To select a specific CUDA version, use the `cuda-version` metapackage (e.g., `conda install -c conda-forge cupy cuda-version=12.0`).

> **Note:** Report any issues with CuPy installed from `conda-forge` to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Use the NVIDIA Container Toolkit to run CuPy container images.

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

## Acknowledgements

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
```