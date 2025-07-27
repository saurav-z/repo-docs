<div align="center">
  <a href="https://ivy.dev/">
    <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
  </a>
</div>

<div align="center">
  <a href="https://github.com/ivy-llc/ivy/stargazers">
    <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/ivy-llc/ivy" alt="GitHub Stars">
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

# Ivy: Seamlessly Convert and Transpile Machine Learning Code Between Frameworks

Ivy empowers you to break down framework barriers by enabling seamless conversion and transpilation of your machine learning models, tools, and libraries between popular frameworks. ([See the original repository](https://github.com/ivy-llc/ivy))

## Key Features

*   **Framework Conversion:** Effortlessly convert code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Code Transpilation:** Utilize the `ivy.transpile` function to convert ML code from one framework to another.
*   **Computational Graph Tracing:** Employ `ivy.trace_graph` to trace and optimize computational graphs.
*   **Easy Installation:**  Simple installation via pip.

## Supported Frameworks

Ivy currently supports conversions between the following frameworks:

<div align="center">
    <a href="https://jax.readthedocs.io">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/jax.svg" alt="JAX Logo">
    </a>
    <a href="https://www.tensorflow.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/tensorflow.svg" alt="TensorFlow Logo">
    </a>
    <a href="https://pytorch.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/pytorch.svg" alt="PyTorch Logo">
    </a>
    <a href="https://numpy.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/numpy.svg" alt="NumPy Logo">
    </a>
</div>

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

## Installation

Install Ivy easily using pip:

```bash
pip install ivy
```

<details>
<summary><b>From Source</b></summary>
<br clear="all" />

Install Ivy from source to access the latest changes:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

## Getting Started with Ivy: Examples

Explore how to use Ivy for framework conversion and graph tracing:

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
  graph = ivy.trace_graph(jax_fn, to="torch", args=(torch_x,))
  ret = graph(torch_x)
  ```
</details>

<details>
  <summary><b>How Ivy Works</b></summary>
  <br clear="all" />

  Ivy's transpiler allows you to use code from any other framework in your own code. For detailed usage, refer to the [full API reference](https://www.docs.ivy.dev/demos/examples_and_demos.html), but the core functions are:

  ```python
  # Converts framework-specific code to a target framework.
  ivy.transpile()

  # Traces a functional graph, removing redundant code.
  ivy.trace_graph()
  ```

  #### `ivy.transpile` Eagerly Transpiles Functions and Classes
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

  #### `ivy.transpile` Lazily Transpiles Modules (Libraries)
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

## Contributing

Join the Ivy community!  Your contributions are welcome. Review the [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and explore our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) for more details.

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