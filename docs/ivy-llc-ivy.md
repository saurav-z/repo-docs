<div style="display: block;" align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>
<br clear="all" />

<div style="margin-top: 10px; margin-bottom: 10px; display: block;" align="center">
    <a href="https://github.com/ivy-llc/ivy/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/ivy-llc/ivy" alt="GitHub stars">
    </a>
    <a href="https://discord.gg/uYRmyPxMQq">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white" alt="Discord">
    </a>
    <a href="https://ivy-llc.github.io/docs/">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/docs-purple" alt="Documentation">
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler">
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests">
    </a>
</div>
<br clear="all" />

# Ivy: Convert and Run Machine Learning Code Across Frameworks

**Ivy is a powerful library that empowers you to seamlessly convert and execute machine learning code between popular frameworks.**

## Key Features

*   **Framework Conversion:** Effortlessly transpile ML models, tools, and libraries between PyTorch, TensorFlow, JAX, and NumPy.
*   **Simplified Cross-Framework Compatibility:** Run your code in the framework of your choice, regardless of its original implementation.
*   **Efficient Graph Tracing:** Trace computational graphs to optimize performance and remove redundant code.
*   **Easy Installation:** Get started quickly with a straightforward `pip install ivy`.
*   **Comprehensive Documentation:** Explore detailed examples, demos, and an API reference.

## Supported Frameworks

Ivy currently supports conversions between the following frameworks:

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

## Installation

Install Ivy using `pip`:

```bash
pip install ivy
```

### From Source

Alternatively, install from source to access the latest updates:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

## Getting Started

### Transpiling Code Between Frameworks

Convert code from one framework to another using `ivy.transpile`.

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

### Tracing a Computational Graph

Trace a computational graph to optimize performance.

```python
import ivy
import torch

def torch_fn(x):
    a = torch.mul(x, x)
    b = torch.mean(x)
    return x * a + b

torch_x = torch.tensor([1., 2., 3.])
graph = ivy.trace_graph(torch_fn, to="torch", args=(torch_x,))
ret = graph(torch_x)
```

### How Ivy Works

Ivy's transpiler enables you to use code from any other framework in your own code. Key functions include:

*   `ivy.transpile()`: Converts framework-specific code to a target framework (see usage in the documentation).
*   `ivy.trace_graph()`: Traces an efficient, fully-functional graph from a function (see usage in the documentation).

#### Eager Transpilation

`ivy.transpile` transpiles eagerly when a class or function is provided:

```python
import ivy
import torch
import tensorflow as tf

def torch_fn(x):
    x = torch.abs(x)
    return torch.sum(x)

x1 = torch.tensor([1., 2.])
x1 = tf.convert_to_tensor([1., 2.])

# Transpilation happens eagerly
tf_fn = ivy.transpile(torch_fn, source="torch", target="tensorflow")

# tf_fn is now tensorflow code and runs efficiently
ret = tf_fn(x1)
```

#### Lazy Transpilation

`ivy.transpile` transpiles lazily if a module (library) is provided:

```python
import ivy
import kornia
import tensorflow as tf

x2 = tf.random.normal((5, 3, 4, 4))

# Module is provided -> transpilation happens lazily
tf_kornia = ivy.transpile(kornia, source="torch", target="tensorflow")

# The transpilation is initialized here, and this function is converted to tensorflow
ret = tf_kornia.color.rgb_to_grayscale(x2)

# Transpilation has already occurred, the tensorflow function runs efficiently
ret = tf_kornia.color.rgb_to_grayscale(x2)
```

## Contribute

Contribute to Ivy and help improve machine learning interoperability!  Check out the [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and learn more in the [Contributing Guide](https://docs.ivy.dev/overview/contributing.html).

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" alt="Contributors"/>
</a>

## Citation

```
@article{lenton2021ivy,
  title={Ivy: Templated deep learning for inter-framework portability},
  author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
  journal={arXiv preprint arXiv:2102.02886},
  year={2021}
}
```

[View the original repository on GitHub](https://github.com/ivy-llc/ivy)