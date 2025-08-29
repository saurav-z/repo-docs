html
<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo"/>
</div>

# CuPy: Accelerate Your NumPy & SciPy Code with GPUs

**CuPy is the go-to library for seamlessly accelerating your NumPy and SciPy code using NVIDIA GPUs and AMD ROCm platforms.** Experience a dramatic performance boost without rewriting your existing Python code.  [Learn more about CuPy on GitHub](https://github.com/cupy/cupy).

[![pypi](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/badge/conda--forge-cupy-blue)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Matrix](https://img.shields.io/matrix/cupy_community:gitter.im?server_fqdn=matrix.org)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)
[![Medium](https://img.shields.io/badge/Medium-CuPy-teal)](https://medium.com/cupy-team)

**Key Features:**

*   **NumPy & SciPy Compatibility:**  CuPy is designed as a drop-in replacement for NumPy and SciPy, making GPU acceleration effortless.
*   **GPU Acceleration:** Leverage the power of NVIDIA CUDA and AMD ROCm GPUs for significantly faster computations.
*   **Easy Integration:**  Simply replace `import numpy as np` with `import cupy as cp` in your existing code.
*   **CUDA & ROCm Support:** Provides access to low-level CUDA and ROCm features, including RawKernels, Streams, and CUDA Runtime APIs for advanced users.
*   **Broad Community Support:** Benefit from an active community and comprehensive documentation.

**Example:**

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

CuPy offers various installation options:

### Pip

Install pre-built binary packages from PyPI:

*   **CUDA 11.x (11.2+):** `pip install cupy-cuda11x`
*   **CUDA 12.x:** `pip install cupy-cuda12x`
*   **CUDA 13.x:** `pip install cupy-cuda13x`
*   **ROCm 4.3 (Experimental):** `pip install cupy-rocm-4-3`
*   **ROCm 5.0 (Experimental):** `pip install cupy-rocm-5-0`

**Note:** For pre-releases, use `--pre -U -f https://pip.cupy.dev/pre`

### Conda

Install from Conda-Forge:

*   **CUDA:** `conda install -c conda-forge cupy`
*   **Slim Installation (without CUDA dependencies):** `conda install -c conda-forge cupy-core`

### Docker

Run CuPy using Docker with the NVIDIA Container Toolkit:

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   [**Website**](https://cupy.dev/)
*   [**Installation Guide**](https://docs.cupy.dev/en/stable/install.html)
*   [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html)
*   [**Examples**](https://github.com/cupy/cupy/tree/main/examples)
*   [**Documentation**](https://docs.cupy.dev/en/stable/)
*   [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
*   [**Forum**](https://groups.google.com/forum/#!forum/cupy)
*   [Release Notes](https://github.com/cupy/cupy/releases)
*   [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   [Contribution Guide](https://docs.cupy.dev/en/stable/contribution.html)

## License

MIT License (see `LICENSE` file).

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