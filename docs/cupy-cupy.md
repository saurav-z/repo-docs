<div align="center">
  <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400" alt="CuPy Logo">
</div>

# CuPy: Accelerate Your Python Code with GPU Power

**CuPy empowers you to harness the computational power of GPUs for lightning-fast array operations, using a familiar NumPy and SciPy interface.**  This allows you to significantly speed up your existing Python code without major rewrites, by leveraging NVIDIA CUDA or AMD ROCm.  Learn more at the [CuPy GitHub Repository](https://github.com/cupy/cupy).

## Key Features

*   **NumPy & SciPy Compatibility:** Seamlessly integrate CuPy into your existing code with a drop-in replacement for NumPy/SciPy functions.
*   **GPU Acceleration:** Execute array operations directly on NVIDIA CUDA or AMD ROCm GPUs for dramatic performance gains.
*   **CUDA & ROCm Integration:** Access low-level CUDA and ROCm features like RawKernels, Streams, and CUDA Runtime APIs for advanced control.
*   **Broad Compatibility:** Supports various CUDA and ROCm versions, including CUDA 11.x, 12.x, and 13.x, as well as ROCm 4.3 and 5.0.
*   **Easy Installation:**  Install quickly using pip or conda, with pre-built binary packages for popular platforms.
*   **Docker Support:** Run CuPy in Docker containers for consistent and reproducible environments.

## Installation

### Pip

Choose the appropriate package for your CUDA or ROCm version:

*   **CUDA 11.x (11.2+):** `pip install cupy-cuda11x`
*   **CUDA 12.x:** `pip install cupy-cuda12x`
*   **CUDA 13.x:** `pip install cupy-cuda13x`
*   **ROCm 4.3 (Experimental):** `pip install cupy-rocm-4-3`
*   **ROCm 5.0 (Experimental):** `pip install cupy-rocm-5-0`

*   **Pre-releases:** `pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre` (replace `cupy-cuda11x` with your target version)

### Conda

Install CuPy from conda-forge:

*   `conda install -c conda-forge cupy`
*   **Slim Installation:** `conda install -c conda-forge cupy-core`
*   **Specific CUDA Version:** `conda install -c conda-forge cupy cuda-version=12.0` (replace `12.0` with your desired version)

### Docker

Run CuPy with the NVIDIA Container Toolkit:

```bash
docker run --gpus all -it cupy/cupy
```

## Resources

*   **Website:** [https://cupy.dev/](https://cupy.dev/)
*   **Documentation:** [https://docs.cupy.dev/en/stable/](https://docs.cupy.dev/en/stable/)
*   **Installation Guide:** [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html)
*   **Tutorial:** [https://docs.cupy.dev/en/stable/user_guide/basic.html](https://docs.cupy.dev/en/stable/user_guide/basic.html)
*   **Examples:** [https://github.com/cupy/cupy/tree/main/examples](https://github.com/cupy/cupy/tree/main/examples)
*   **API Reference:** [https://docs.cupy.dev/en/stable/reference/](https://docs.cupy.dev/en/stable/reference/)
*   **Release Notes:** [https://github.com/cupy/cupy/releases](https://github.com/cupy/cupy/releases)
*   **Projects using CuPy:** [https://github.com/cupy/cupy/wiki/Projects-using-CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)
*   **Contribution Guide:** [https://docs.cupy.dev/en/stable/contribution.html](https://docs.cupy.dev/en/stable/contribution.html)
*   **Forum:** [https://groups.google.com/forum/#!forum/cupy](https://groups.google.com/forum/#!forum/cupy)

## License

CuPy is licensed under the MIT License (see `LICENSE` file).

## Reference

Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C. (2017). CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. *Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*. [[PDF](http://learningsys.org/nips17/assets/papers/paper_16.pdf)]

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```