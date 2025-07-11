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

# Ivy: Seamlessly Convert Machine Learning Code Between Frameworks

Ivy empowers you to effortlessly convert and utilize machine learning models and libraries across various frameworks.  [Explore the Ivy repository on GitHub](https://github.com/ivy-llc/ivy).

## Key Features

*   **Framework Conversion:** Transpile code between popular ML frameworks like PyTorch, TensorFlow, JAX, and NumPy.
*   **Easy Installation:**  Get started quickly with a simple `pip install ivy`.
*   **Code Tracing:** Trace and optimize computational graphs for enhanced performance.
*   **Flexible Transpilation:** Choose between eager and lazy transpilation based on your needs.
*   **Extensive Examples:**  Explore a wide range of demos and tutorials in the [examples page](https://www.docs.ivy.dev/demos/examples_and_demos.html).

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
<summary><b>From Source</b></summary>
<br clear="all" />

Install from source to access the latest features:

```bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

<br clear="all" />

## Supported Frameworks

| Framework  | Source | Target |
|------------|:------:|:------:|
| PyTorch    |   ✅   |   🚧   |
| TensorFlow |   🚧   |   ✅   |
| JAX        |   🚧   |   ✅   |
| NumPy      |   🚧   |   ✅   |

## Getting Started with Ivy

```python
import ivy
import torch
import tensorflow as tf

# Transpile PyTorch code to TensorFlow
def torch_fn(x):
    a = torch.mul(x, x)
    b = torch.mean(x)
    return x * a + b

tf_fn = ivy.transpile(torch_fn, source="torch", target="tensorflow")

# Use the transpiled TensorFlow function
tf_x = tf.convert_to_tensor([1., 2., 3.])
ret = tf_fn(tf_x)
```

```python
import ivy
import torch

# Trace a computational graph for a PyTorch function
def torch_fn(x):
    a = torch.mul(x, x)
    b = torch.mean(x)
    return x * a + b

torch_x = torch.tensor([1., 2., 3.])
graph = ivy.trace_graph(torch_fn, to="torch", args=(torch_x,))
ret = graph(torch_x)
```

<details>
<summary><b>How Ivy Works</b></summary>
<br clear="all" />

Ivy's transpiler allows you to utilize code from any framework within your own projects.

Key functions:

*   `ivy.transpile()`: Converts framework-specific code to your desired target framework.  Refer to the documentation for usage details.
*   `ivy.trace_graph()`: Creates an efficient, fully-functional graph from a function, streamlining the code. Explore the documentation for detailed usage information.

### Transpilation Behavior

*   **Eager Transpilation:** When a class or function is provided to `ivy.transpile`, the code is transpiled immediately.

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

# tf_fn now contains TensorFlow code
ret = tf_fn(x1)
```

*   **Lazy Transpilation:** If a module (library) is provided, transpilation happens lazily.

```python
import ivy
import kornia
import tensorflow as tf

x2 = tf.random.normal((5, 3, 4, 4))

# Transpilation is lazy when a module is provided
tf_kornia = ivy.transpile(kornia, source="torch", target="tensorflow")

# Transpilation occurs when this function is called
ret = tf_kornia.color.rgb_to_grayscale(x2)

# The TensorFlow function executes efficiently
ret = tf_kornia.color.rgb_to_grayscale(x2)
```

</details>

<br clear="all" />

## Contributing

We value community contributions!  Contribute code, fix bugs, or provide feedback to improve Ivy.

Check out our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html) and [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) for more information.

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
```

Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords like "machine learning," "framework conversion," "PyTorch," "TensorFlow," "JAX," and "NumPy" in the headings and descriptions.
*   **Clear Structure:**  Uses headings (H2 and H3) to organize the content for readability and SEO.
*   **Concise Hook:** Starts with a compelling one-sentence summary of Ivy's purpose.
*   **Bulleted Key Features:**  Provides a clear overview of Ivy's capabilities.
*   **Clear Examples:**  Maintains the original examples, but with better formatting.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Internal Links:**  Uses internal links (e.g., to the contributing guide, open tasks, and examples page).
*   **Concise Explanations:**  Streamlined explanations and removed unnecessary wording.
*   **Call to Action:**  Encourages users to explore the documentation and contribute.
*   **GitHub Link:** Added a direct link back to the original GitHub repository at the start.
*   **Markdown Formatting:** Improved markdown formatting for better readability on GitHub.
*   **Clearer "How Ivy Works" section:**  Improved the "How does ivy work?" section to be more concise and easier to understand.  Reformatted the examples within this section for better readability and clarity.