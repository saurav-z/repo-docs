<div style="display: block;" align="center">
    <a href="https://ivy.dev/">
        <img class="dark-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy Logo"/>
    </a>
</div>
<br clear="all" />

<div style="margin-top: 10px; margin-bottom: 10px; display: block;" align="center">
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
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests"/>
    </a>
</div>
<br clear="all" />

## Ivy: Seamlessly Convert Machine Learning Code Between Frameworks

Ivy empowers you to effortlessly convert and utilize machine learning code across different frameworks, simplifying your workflow and boosting your productivity.  [Explore the Ivy repository on GitHub](https://github.com/ivy-llc/ivy).

**Key Features:**

*   **Framework Conversion:** Easily convert ML models and libraries between PyTorch, TensorFlow, JAX, and NumPy.
*   **Code Portability:**  Write code once and run it across multiple frameworks, maximizing flexibility.
*   **Efficient Graph Tracing:** Trace computational graphs for optimized performance and streamlined execution.
*   **Simplified Workflow:** Reduce the complexities of framework-specific code, saving time and effort.

<div style="display: block;" align="center">
    <div>
    <a href="https://jax.readthedocs.io">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/jax.svg" alt="JAX"/>
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <a href="https://www.tensorflow.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/tensorflow.svg" alt="TensorFlow"/>
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <a href="https://pytorch.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/pytorch.svg" alt="PyTorch"/>
    </a>
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <img class="dark-light" width="5%" src="https://github.com/ivy-llc/assets/blob/main/assets/empty.png?raw=true" alt="Spacer">
    <a href="https://numpy.org">
        <img class="dark-light" width="100" height="100" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/numpy.svg" alt="NumPy"/>
    </a>
    </div>
</div>

<br clear="all" />

## Installation

Install Ivy with a single command using **pip**:

```bash
pip install ivy
```

<details>
<summary><b>From Source</b></summary>
<br clear="all" />

To install the latest version from source:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

## Supported Frameworks

Ivy currently supports conversions between these popular machine learning frameworks:

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

<br clear="all" />

## Getting Started with Ivy

Explore these examples to see Ivy in action!  For more in-depth tutorials and demos, visit the [examples page](https://www.docs.ivy.dev/demos/examples_and_demos.html) in the documentation.

  <details>
    <summary><b>Transpiling code from one framework to another</b></summary>
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
    <summary><b>Tracing a computational graph of any code</b></summary>
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

Ivy's transpiler makes it easy to use code from different frameworks within your own project.  Here are the key functions:

```python
# Converts framework-specific code to a target framework. See usage in the documentation
ivy.transpile()

# Creates an efficient and fully-functional graph from a function. See usage in the documentation
ivy.trace_graph()
```

#### `ivy.transpile` Eagerly Transpiles for Functions and Classes

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

#### `ivy.transpile` Lazily Transpiles for Modules (Libraries)

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

Ivy welcomes contributions from everyone!  Whether you're writing code, finding bugs, or providing feedback, your input is valuable.

Browse our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) to find ways to contribute and learn more in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html).

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