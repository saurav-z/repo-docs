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

## Ivy: Effortlessly Convert ML Code Between Frameworks

**Ivy is a groundbreaking library that empowers you to seamlessly convert machine learning models and code between frameworks, fostering unprecedented flexibility and interoperability.**  [Explore the original repository on GitHub](https://github.com/ivy-llc/ivy).

### Key Features

*   **Framework Conversion:** Easily transpile code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Simplified Model Porting:**  Quickly move models across different ML frameworks.
*   **Code Reusability:**  Maximize code reuse by leveraging the same code across various frameworks.
*   **Computational Graph Tracing:** Trace and optimize computational graphs for improved performance.
*   **Flexible Transpilation:** Eager and lazy transpilation options to suit your needs.

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

### Installation

Install Ivy using pip:

```bash
pip install ivy
```

<details>
<summary><b>From Source</b></summary>
<br clear="all" />

Install from source to access the latest changes:

``` bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

### Supported Frameworks

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

<br clear="all" />

### Getting Started

Here are some examples to get you started with Ivy: The [examples page](https://www.docs.ivy.dev/demos/examples_and_demos.html) features more demos and tutorials.

  <details>
    <summary><b>Transpiling Code</b></summary>
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
    <summary><b>Tracing a Computational Graph</b></summary>
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
<summary><b>How Ivy Works</b></summary>
<br clear="all" />

Ivy's transpiler enables you to use code from any other framework in your own code.

``` python
# Converts framework-specific code to a target framework of choice.
ivy.transpile()

# Traces an efficient fully-functional graph from a function.
ivy.trace_graph()
```

#### `ivy.transpile` Eagerly Transpiles Functions/Classes:

``` python
import ivy
import torch
import tensorflow as tf

def torch_fn(x):
    x = torch.abs(x)
    return torch.sum(x)

x1 = torch.tensor([1., 2.])
x1 = tf.convert_to_tensor([1., 2.])

# Eager Transpilation
tf_fn = ivy.transpile(test_fn, source="torch", target="tensorflow")

# tf_fn is now tensorflow code
ret = tf_fn(x1)
```

#### `ivy.transpile` Lazily Transpiles Modules:

``` python
import ivy
import kornia
import tensorflow as tf

x2 = tf.random.normal((5, 3, 4, 4))

# Lazy Transpilation
tf_kornia = ivy.transpile(kornia, source="torch", target="tensorflow")

# Transpilation happens here
ret = tf_kornia.color.rgb_to_grayscale(x2)

# Function runs efficiently
ret = tf_kornia.color.rgb_to_grayscale(x2)
```
</details>

<br clear="all" />

### Contributing

Your contributions are welcome! [Explore Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and learn more in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html).

<br clear="all" />

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" alt="Contributors"/>
</a>

<br clear="all" />
<br clear="all" />

### Citation

```
@article{lenton2021ivy,
  title={Ivy: Templated deep learning for inter-framework portability},
  author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
  journal={arXiv preprint arXiv:2102.02886},
  year={2021}
}
```