html
<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a small, fully-featured deep learning framework designed for simplicity and ease of use.** It sits between PyTorch and micrograd, offering a streamlined approach to building and running neural networks. Learn more at the original repository:  <a href="https://github.com/tinygrad/tinygrad">tinygrad on GitHub</a>.

Key Features:

*   **Runs LLaMA and Stable Diffusion:** Experiment with cutting-edge AI models directly within tinygrad.
*   **Lazy Evaluation:** Experience automatic kernel fusion for optimized performance, demonstrated through simple matmul examples.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, a tensor library, optimizers, and dataloaders.
*   **Wide Accelerator Support:** Enjoy out-of-the-box support for GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU. Adding new accelerators is straightforward, requiring support for only ~25 low-level ops.
*   **Easy Installation:** Install from source or directly via pip.

---

## Getting Started

### Installation

Install tinygrad using either of the following methods:

#### From Source

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

#### Direct (master)

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Documentation and Examples

*   Explore detailed documentation and a quick start guide on the [docs website](https://docs.tinygrad.org/).
*   For a quick comparison with PyTorch, see the provided code example.

## Contributing

tinygrad welcomes contributions!  Please review the guidelines to ensure your pull request is accepted.

### Running Tests

Install pre-commit hooks with `pre-commit install` to automatically run the linter, mypy, and tests. Additional testing examples are available in the [.github/workflows/test.yml](.github/workflows/test.yml) file.

```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process replay tests

Ensure your refactor or speedup PRs include [pr] in the pull request title and run the appropriate [Process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) tests for comparisons.