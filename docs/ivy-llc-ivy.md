<div style="display: block;" align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>
<br clear="all" />

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
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests"/>
    </a>
</div>
<br clear="all" />

# Ivy: Seamlessly Convert and Transpile Machine Learning Code Between Frameworks

Ivy simplifies machine learning by enabling effortless conversion and transpilation of your models and code across popular frameworks. [Explore the Ivy repository on GitHub](https://github.com/ivy-llc/ivy) for more details.

## Key Features

*   **Framework Conversion:** Easily convert ML models and code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Code Transpilation:** Use `ivy.transpile` to convert code from one framework to another.
*   **Computational Graph Tracing:** Trace computational graphs with `ivy.trace_graph` for optimization and analysis.
*   **Flexible Usage:** Supports transpilation of both functions/classes and entire modules.
*   **Open Source:** Ivy is an open-source library, welcoming community contributions.

## Supported Frameworks

Ivy currently supports conversions between the following frameworks:

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

<div style="display: block;" align="center">
    <div>
    <a href="https://jax.readthedocs.io">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/jax.svg" alt="JAX Logo">
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://www.tensorflow.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/tensorflow.svg" alt="TensorFlow Logo">
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://pytorch.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/pytorch.svg" alt="PyTorch Logo">
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="">
    <a href="https://numpy.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/numpy.svg" alt="NumPy Logo">
    </a>
    </div>
</div>

<br clear="all" />

## Installation

Install Ivy using pip:

```bash
pip install ivy
```

<details>
<summary><b>Installation from Source</b></summary>
<br clear="all" />

Install Ivy from source to take advantage of the latest changes:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

## Getting Started with Ivy: Examples

Here are some examples to illustrate how to use Ivy.  For more detailed examples and use cases, visit the [Ivy Demos and Examples](https://www.docs.ivy.dev/demos/examples_and_demos.html) page.

  <details>
    <summary><b>Transpiling code between frameworks</b></summary>
    <br clear="all" />

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
    <summary><b>Tracing a computational graph</b></summary>
    <br clear="all" />

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
<br clear="all" />

Ivy\'s transpiler lets you use code from any framework in your own code.  Key functions to use:

```python
# Converts framework-specific code to a target framework.  See the docs for usage.
ivy.transpile()

# Traces an efficient fully-functional graph, removing redundant code. See the docs for usage.
ivy.trace_graph()
```

### `ivy.transpile`

#### Eager Transpilation with Functions and Classes

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

#### Lazy Transpilation with Modules (Libraries)

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
</details>

<br clear="all" />

## Contributing

Contribute to the growth of Ivy!  We welcome all contributions, including code, bug fixes, and feedback.

Check out the [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and the [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) in the docs.

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