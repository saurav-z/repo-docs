<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a tiny, fully-featured deep learning framework, designed for simplicity and ease of use, making it ideal for experimenting with new accelerators and understanding the inner workings of deep learning.**  Explore the power of tinygrad and [contribute on GitHub](https://github.com/tinygrad/tinygrad)!

### Key Features:

*   **Lightweight and Accessible:**  A small codebase makes it easy to understand, modify, and extend.
*   **Accelerated Performance:** Supports a wide range of accelerators including GPU (OpenCL, METAL, CUDA, AMD, NV, QCOM, WEBGPU), CPU (C Code, LLVM).
*   **Easy Accelerator Integration:**  Adding support for new hardware is straightforward, requiring only implementation of ~25 low-level operations.
*   **Full-Featured:**  Capable of running complex models like LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Smartly fuses operations into efficient kernels, maximizing performance.
*   **Simple Neural Network Creation:** Includes built-in autograd, tensor library, optimizer, and data loader functionality.
*   **Active Development:** Supported by [tiny corp](https://tinygrad.org) with ongoing improvements and community contributions.

### Key Benefits
*   **Understandability:** Tinygrad allows a deeper understanding of Deep Learning frameworks and their underlying structure.
*   **Rapid Prototyping:** The framework is flexible and easily modified which provides for faster prototyping.
*   **Community Driven:** Tinygrad is an open-source, community driven project

### Examples:

*   **LLaMA and Stable Diffusion:** Run cutting-edge models with ease.
*   **Lazy Matrix Multiplication:** Witness the power of lazy evaluation with this concise example:

    ```bash
    DEBUG=3 python3 -c "from tinygrad import Tensor; N = 1024; a, b = Tensor.rand(N, N), Tensor.rand(N, N); c = (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2); print((c.numpy() - (a.numpy() @ b.numpy())).mean())"
    ```

*   **Simple Neural Network:** Build and train a basic neural network in just a few lines:

    ```python
    from tinygrad import Tensor, nn

    class LinearNet:
      def __init__(self):
        self.l1 = Tensor.kaiming_uniform(784, 128)
        self.l2 = Tensor.kaiming_uniform(128, 10)
      def __call__(self, x:Tensor) -> Tensor:
        return x.flatten(1).dot(self.l1).relu().dot(self.l2)

    model = LinearNet()
    optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

    x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

    with Tensor.train():
      for i in range(10):
        optim.zero_grad()
        loss = model(x).sparse_categorical_crossentropy(y).backward()
        optim.step()
        print(i, loss.item())
    ```

### Installation:

1.  **From Source:**
    ```bash
    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    python3 -m pip install -e .
    ```

2.  **Direct (master):**
    ```bash
    python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
    ```

### Documentation and Resources:

*   **Homepage:** [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
*   **Documentation:** [https://docs.tinygrad.org/](https://docs.tinygrad.org/)
*   **Discord:** [https://discord.gg/ZjZadyC7PK](https://discord.gg/ZjZadyC7PK)

### Contributing:

We welcome contributions! Please review the [contribution guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) before submitting a pull request.

### Running Tests

```sh
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```