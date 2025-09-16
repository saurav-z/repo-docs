<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: Accelerate Your Python with GPU-Powered NumPy & SciPy

**CuPy empowers you to leverage the power of GPUs for accelerated numerical computing, offering a seamless NumPy/SciPy experience on NVIDIA CUDA and AMD ROCm platforms.**  Access the original repository [here](https://github.com/cupy/cupy).

## Key Features

*   **NumPy & SciPy Compatibility:** CuPy provides a drop-in replacement for NumPy and SciPy, allowing you to run your existing code with minimal modifications.
*   **GPU Acceleration:**  Significantly speeds up computations by executing them on NVIDIA GPUs or AMD ROCm platforms.
*   **CUDA and ROCm Support:**  Works seamlessly with NVIDIA CUDA and AMD ROCm environments.
*   **Low-Level CUDA Access:** Offers advanced features like RawKernels, Streams, and direct access to CUDA Runtime APIs for fine-grained control and optimization.

## Installation

### Pip

Install pre-built binary packages (wheels) from PyPI for various CUDA and ROCm versions.

| Platform              | Architecture      | Command                                                       |
| --------------------- | ----------------- | ------------------------------------------------------------- |
| CUDA 11.x (11.2+)     | x86_64 / aarch64  | `pip install cupy-cuda11x`                                    |
| CUDA 12.x             | x86_64 / aarch64  | `pip install cupy-cuda12x`                                    |
| CUDA 13.x             | x86_64 / aarch64  | `pip install cupy-cuda13x`                                    |
| ROCm 4.3 (Experimental)          | x86_64            | `pip install cupy-rocm-4-3`                                   |
| ROCm 5.0 (Experimental)          | x86_64            | `pip install cupy-rocm-5-0`                                   |

> **Note:** For pre-releases, use `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre`.  See the [Installation Guide](https://docs.cupy.dev/en/stable/install.html) for detailed instructions.

### Conda

Install with Conda from Conda-Forge:

| Platform              | Architecture                | Command                                                       |
| --------------------- | --------------------------- | ------------------------------------------------------------- |
| CUDA                  | x86_64 / aarch64 / ppc64le  | `conda install -c conda-forge cupy`                           |

For a slim installation without CUDA dependencies, use `conda install -c conda-forge cupy-core`. To specify a CUDA version (e.g., 12.0), use `conda install -c conda-forge cupy cuda-version=12.0`. Report any Conda-Forge issues to [cupy-feedstock](https://github.com/conda-forge/cupy-feedstock/issues).

### Docker

Run CuPy containers using the NVIDIA Container Toolkit:

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   [**Website**](https://cupy.dev/)
*   [**Installation Guide**](https://docs.cupy.dev/en/stable/install.html) - Build from source instructions
*   [**Release Notes**](https://github.com/cupy/cupy/releases)
*   [**Projects using CuPy**](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [**Contribution Guide**](https://docs.cupy.dev/en/stable/contribution.html)
*   [**GPU Acceleration in Python using CuPy and Numba (GTC November 2021 Technical Session)**](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31149/)
*   [**GPU-Acceleration of Signal Processing Workflows using CuPy and cuSignal[^1] (ICASSP'21 Tutorial)**](https://github.com/awthomp/cusignal-icassp-tutorial)
*   [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html)
*   [**Examples**](https://github.com/cupy/cupy/tree/main/examples)
*   [**Documentation**](https://docs.cupy.dev/en/stable/)
*   [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
*   [**Forum**](https://groups.google.com/forum/#!forum/cupy)

[^1]: cuSignal is now part of CuPy starting v13.0.0.

## License

MIT License (see `LICENSE` file).

CuPy is designed based on NumPy's API and SciPy's API (see `docs/source/license.rst` file).

## Development

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