<div align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>
<br clear="all" />

<div align="center">
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
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Transpiler Tests">
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests">
    </a>
</div>
<br clear="all" />

# Ivy: Effortlessly Convert Machine Learning Code Between Frameworks

**Ivy is a powerful tool that allows you to seamlessly convert and use machine learning code across different frameworks, promoting code reusability and reducing vendor lock-in.** Access the original repository here: [https://github.com/ivy-llc/ivy](https://github.com/ivy-llc/ivy)

## Key Features

*   **Framework Conversion:** Convert ML models, tools, and libraries between popular frameworks.
*   **Simplified Development:** Write code once and run it across different ML platforms.
*   **Improved Portability:** Easily migrate your ML projects between frameworks.
*   **Efficient Graph Tracing:** Trace and optimize computational graphs for improved performance.

## Supported Frameworks

Ivy currently supports conversions between the following frameworks:

| Framework    | Source | Target |
|--------------|:------:|:------:|
| PyTorch      |   ✅   |   🚧   |
| TensorFlow   |   🚧   |   ✅   |
| JAX          |   🚧   |   ✅   |
| NumPy        |   🚧   |   ✅   |

<div align="center">
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

To install from source, clone the repository and install it using:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

## Using Ivy

Here are some examples to get you started:

<details>
    <summary><b>Transpiling any code from one framework to another</b></summary>
    <br clear="all" />

   ``` python
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
    <summary><b>Tracing a computational graph of any code</b></summary>
    <br clear="all" />

   ``` python
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
<summary><b>How does ivy work?</b></summary>
<br clear="all" />

Ivy\'s transpiler allows you to use code from any other framework in your own code.
Feel free to head over to the docs for the full API
reference, but the functions you\'d most likely want to use are:

``` python
# Converts framework-specific code to a target framework of choice. See usage in the documentation
ivy.transpile()

# Traces an efficient fully-functional graph from a function, removing all wrapping and redundant code. See usage in the documentation
ivy.trace_graph()
```

#### `ivy.transpile` will eagerly transpile if a class or function is provided

``` python
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

``` python
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

We welcome contributions from everyone!  You can contribute by writing code, fixing bugs, or providing feedback.  Explore our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and learn more in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html)

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