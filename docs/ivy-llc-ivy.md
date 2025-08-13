<div align="center">
  <a href="https://ivy.dev/">
    <img src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo" width="50%">
  </a>
</div>

<div align="center">
  <a href="https://github.com/ivy-llc/ivy/stargazers">
    <img src="https://img.shields.io/github/stars/ivy-llc/ivy?style=social" alt="GitHub Stars">
  </a>
  <a href="https://discord.gg/uYRmyPxMQq">
    <img src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="https://ivy-llc.github.io/docs/">
    <img src="https://img.shields.io/badge/docs-purple" alt="Documentation">
  </a>
  <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
    <img src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler">
  </a>
  <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
    <img src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests">
  </a>
</div>

<br>

## Ivy: Seamlessly Convert and Utilize Machine Learning Code Across Frameworks

Ivy is your go-to solution for eliminating framework lock-in and maximizing code portability in the world of machine learning. ([View on GitHub](https://github.com/ivy-llc/ivy))

### Key Features:

*   **Framework Conversion:** Effortlessly convert ML models, tools, and libraries between popular frameworks.
*   **Supported Frameworks:**
    *   PyTorch
    *   TensorFlow
    *   JAX
    *   NumPy
*   **Code Tracing:** Trace computational graphs for efficient execution and analysis.
*   **Easy Installation:** Simple installation via pip.
*   **Comprehensive Documentation:** Access detailed documentation and examples to get started quickly.

<div align="center">
  <div>
    <a href="https://jax.readthedocs.io">
      <img width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/jax.svg" alt="JAX">
    </a>
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://www.tensorflow.org">
      <img width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/tensorflow.svg" alt="TensorFlow">
    </a>
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://pytorch.org">
      <img width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/pytorch.svg" alt="PyTorch">
    </a>
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://numpy.org">
      <img width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/numpy.svg" alt="NumPy">
    </a>
  </div>
</div>

<br>

### Installation

Install Ivy using pip:

```bash
pip install ivy
```

#### From Source

Alternatively, install from source to access the latest updates:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

<br>

### Framework Compatibility

Ivy currently supports converting code between the following frameworks:

| Framework    | Source | Target |
|--------------|:------:|:------:|
| PyTorch      |   ✅   |   🚧   |
| TensorFlow   |   🚧   |   ✅   |
| JAX          |   🚧   |   ✅   |
| NumPy        |   🚧   |   ✅   |

<br>

### Getting Started with Ivy: Examples

  <details>
    <summary><b>Transpiling Code Between Frameworks</b></summary>
    <br>

   ```python
   import ivy
   import torch
   import tensorflow as tf

   def torch_fn(x):
       a = torch.mul(x, x)
       b = torch.mean(x)
       return x * a + b

   tf_fn = ivy.transpile(torch_fn, source="torch", target="tensorflow")

   tf_x = tf.convert_to_tensor([1., 2., 3.])
   ret = tf_fn(tf_x)
   ```

  </details>

  <details>
    <summary><b>Tracing Computational Graphs</b></summary>
    <br>

   ```python
   import ivy
   import torch

   def torch_fn(x):
       a = torch.mul(x, x)
       b = torch.mean(x)
       return x * a + b

   torch_x = torch.tensor([1., 2., 3.])
   graph = ivy.trace_graph(jax_fn, to="torch", args=(torch_x,))
   ret = graph(torch_x)
   ```

   </details>

<details>
<summary><b>How Ivy Works</b></summary>
<br>

Ivy's transpiler enables you to seamlessly integrate code from different frameworks into your projects. Core functions include:

```python
# Converts framework-specific code to a target framework. See documentation for usage.
ivy.transpile()

# Creates an efficient graph from a function, removing redundant code.  See documentation for usage.
ivy.trace_graph()
```

##### `ivy.transpile` with functions/classes

```python
import ivy
import torch
import tensorflow as tf

def torch_fn(x):
    x = torch.abs(x)
    return torch.sum(x)

x1 = torch.tensor([1., 2.])
x1 = tf.convert_to_tensor([1., 2.])

# Eager transpilation
tf_fn = ivy.transpile(test_fn, source="torch", target="tensorflow")

# tf_fn now contains tensorflow code, running efficiently
ret = tf_fn(x1)
```

##### `ivy.transpile` with modules (libraries)

```python
import ivy
import kornia
import tensorflow as tf

x2 = tf.random.normal((5, 3, 4, 4))

# Module provided -> lazy transpilation
tf_kornia = ivy.transpile(kornia, source="torch", target="tensorflow")

# Transpilation initiated here. The function is converted to TensorFlow
ret = tf_kornia.color.rgb_to_grayscale(x2)

# Transpilation complete.  The TensorFlow function runs efficiently
ret = tf_kornia.color.rgb_to_grayscale(x2)
```
</details>

<br>

### Contribute to Ivy

Join the community and help shape the future of framework-agnostic machine learning! Check out the [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and the [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) for details.

<br>

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" alt="Contributors">
</a>

<br>
<br>

### Citation

```
@article{lenton2021ivy,
  title={Ivy: Templated deep learning for inter-framework portability},
  author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
  journal={arXiv preprint arXiv:2102.02886},
  year={2021}
}
```