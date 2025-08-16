<div align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>
<br clear="all" />

<div align="center">
    <a href="https://github.com/ivy-llc/ivy/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/ivy-llc/ivy" alt="GitHub Stars"/>
    </a>
    <a href="https://discord.gg/uYRmyPxMQq">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white" alt="Discord"/>
    </a>
    <a href="https://ivy-llc.github.io/docs/">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/docs-purple" alt="Documentation"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler Status"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests Status"/>
    </a>
</div>
<br clear="all" />

# Ivy: Seamlessly Convert Machine Learning Code Between Frameworks

**Ivy simplifies machine learning by enabling effortless conversion of models and code between frameworks like PyTorch, TensorFlow, and JAX.**  Visit the [original repo](https://github.com/ivy-llc/ivy) to learn more.

## Key Features

*   **Framework Conversion:** Easily transpile your ML code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Simplified Development:** Use code from any supported framework within your preferred environment.
*   **Code Tracing:** Trace computational graphs for efficient execution.
*   **Easy Installation:** Quick setup with `pip install ivy`.
*   **Comprehensive Documentation:** Explore detailed examples and API reference in the [Ivy documentation](https://ivy-llc.github.io/docs/).

## Supported Frameworks

Ivy supports conversion to and from the following frameworks:

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

<br clear="all" />

## Getting Started

### Installation

Install Ivy using `pip`:

```bash
pip install ivy
```

### From Source

For the latest changes, install from source:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

### Examples

#### Transpiling Code Between Frameworks

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

#### Tracing a Computational Graph

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

<br clear="all" />

## How Ivy Works

Ivy's transpiler enables you to utilize code from different frameworks seamlessly within your own projects. Key functions include:

*   **`ivy.transpile()`:** Converts framework-specific code to your target framework.
*   **`ivy.trace_graph()`:** Creates efficient computational graphs from functions, streamlining execution.

**Eager Transpilation:** `ivy.transpile` eagerly transpiles when provided with a class or function.

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
tf_fn = ivy.transpile(test_fn, source="torch", target="tensorflow")

# tf_fn is now tensorflow code and runs efficiently
ret = tf_fn(x1)
```

**Lazy Transpilation:** `ivy.transpile` lazily transpiles when a module (library) is provided.

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

## Contributing

Contribute to Ivy and help make a difference!  Check out our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) for more information.

<br clear="all" />

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" alt="Contributors"/>
</a>

<br clear="all" />
<br clear="all" />

## Citation

```
@article{lenton2021ivy,
  title={Ivy: Templated deep learning for inter-framework portability},
  author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
  journal={arXiv preprint arXiv:2102.02886},
  year={2021}
}
```