<div align="center">
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
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white" alt="Discord badge"/>
    </a>
    <a href="https://ivy-llc.github.io/docs/">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/docs-purple" alt="Documentation"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler Workflow Status"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests Workflow Status"/>
    </a>
</div>
<br clear="all" />

# Ivy: Effortlessly Convert and Run Machine Learning Code Between Frameworks

Ivy simplifies machine learning by enabling seamless conversion and execution of models across different frameworks. Explore the [Ivy repository](https://github.com/ivy-llc/ivy) to get started!

## Key Features

*   **Framework Conversion:** Easily convert code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Code Transpilation:**  Utilize `ivy.transpile` to convert ML models, tools, and libraries from one framework to another.
*   **Computational Graph Tracing:**  Trace computational graphs to understand and optimize your models.
*   **Flexible Usage:** Transpile individual functions/classes or entire modules.

<div align="center">
    <div>
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/jax.svg" alt="JAX"/>
        </a>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/tensorflow.svg" alt="TensorFlow"/>
        </a>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <a href="https://pytorch.org">
            <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/pytorch.svg" alt="PyTorch"/>
        </a>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt=""/>
        <a href="https://numpy.org">
            <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/numpy.svg" alt="NumPy"/>
        </a>
    </div>
</div>

<br clear="all" />

## Installation

Install Ivy easily using pip:

```bash
pip install ivy
```

<details>
    <summary><b>From Source</b></summary>
    <br clear="all" />

You can also install from source:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

## Supported Frameworks

Ivy supports conversions between the following frameworks:

| Framework    | Source | Target |
|--------------|:------:|:------:|
| PyTorch      |   ✅   |   🚧   |
| TensorFlow   |   🚧   |   ✅   |
| JAX          |   🚧   |   ✅   |
| NumPy        |   🚧   |   ✅   |

<br clear="all" />

## Getting Started with Ivy

Here's how to begin using Ivy:

<details>
    <summary><b>Transpiling Code Between Frameworks</b></summary>
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
    <summary><b>Tracing a Computational Graph</b></summary>
    <br clear="all" />

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
</details>

<details>
    <summary><b>How does Ivy work?</b></summary>
    <br clear="all" />

Ivy's transpiler allows you to use code from any other framework in your own code.
For full API reference, check out the [docs](https://www.docs.ivy.dev/), but here are the main functions:

```python
# Converts framework-specific code to a target framework of choice. See usage in the documentation
ivy.transpile()

# Traces an efficient fully-functional graph from a function, removing all wrapping and redundant code. See usage in the documentation
ivy.trace_graph()
```

#### `ivy.transpile` will eagerly transpile if a class or function is provided

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

#### `ivy.transpile` will lazily transpile if a module (library) is provided

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

## Contribute

Contribute to Ivy and help make a difference!  Your contributions, whether in code, bug fixes, or feedback, are welcome.
Check out our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) for more details.

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