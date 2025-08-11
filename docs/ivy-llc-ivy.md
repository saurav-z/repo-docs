<div style="display: block;" align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>

<div style="margin-top: 10px; margin-bottom: 10px; display: block;" align="center">
    <a href="https://github.com/ivy-llc/ivy/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/ivy-llc/ivy" alt="GitHub stars"/>
    </a>
    <a href="https://discord.gg/uYRmyPxMQq">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white" alt="Discord"/>
    </a>
    <a href="https://ivy-llc.github.io/docs/">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/docs-purple" alt="Documentation"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler Workflow"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests Workflow"/>
    </a>
</div>

# Ivy: Seamlessly Convert ML Code Between Frameworks

**Ivy empowers you to break down framework barriers, enabling effortless model and code conversion between popular machine learning frameworks like TensorFlow, PyTorch, JAX, and NumPy - [Explore the code on GitHub!](https://github.com/ivy-llc/ivy)**

## Key Features

*   **Framework Conversion:** Easily transpile your ML code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Simplified Portability:**  Eliminate the need to rewrite code for different frameworks.
*   **Efficient Tracing:**  Create optimized computational graphs.
*   **Eager and Lazy Transpilation:** Supports both eager and lazy transpilation based on the input type (function/class or module).
*   **Comprehensive Documentation:** Detailed API reference available [here](https://ivy-llc.github.io/docs/).

## Supported Frameworks

Ivy currently supports conversions to and from the following frameworks:

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

## Installation

Install Ivy using pip:

```bash
pip install ivy
```

**From Source (for latest changes):**

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

## Getting Started

### Transpiling Code

Here's an example of how to transpile PyTorch code to TensorFlow:

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

### Tracing Computational Graphs

Trace an efficient graph:

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

## How Ivy Works

Ivy's transpiler allows you to use code from any other framework in your own code.

*   **`ivy.transpile()`**: Converts framework-specific code to your desired target framework.
*   **`ivy.trace_graph()`**: Traces an efficient, fully-functional graph from a function.

### Transpiling Behavior

*   **Eager Transpilation (Functions/Classes):**  `ivy.transpile` transpiles immediately when a function or class is provided.

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

*   **Lazy Transpilation (Modules):**  Transpilation occurs when a module (library) is provided, allowing for on-demand conversion.

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

We welcome contributions!  Find open tasks and guidelines in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) and on the [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) page.

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