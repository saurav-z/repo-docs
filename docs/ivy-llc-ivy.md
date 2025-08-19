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
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/docs-purple" alt="Documentation badge"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test transpiler badge"/>
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration tests badge"/>
    </a>
</div>
<br clear="all" />

# Ivy: Unleash Inter-Framework ML Code Conversion

Ivy is a powerful library that simplifies machine learning by enabling seamless conversion of models and code between popular frameworks. ([See the original repo](https://github.com/ivy-llc/ivy))

## Key Features

*   **Framework Conversion:** Transpile your machine learning code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Graph Tracing:** Trace and optimize computational graphs for performance and portability.
*   **Easy Installation:** Simple installation via pip.
*   **Comprehensive Documentation:**  Extensive documentation and examples to help you get started.
*   **Open Source:**  Actively developed and open to community contributions.

## Supported Frameworks

Ivy currently supports conversions between these frameworks:

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

You can also install Ivy from source to access the latest features:

``` bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

</details>

## Getting Started with Ivy

Here's how to use Ivy to convert your code between different ML frameworks.  For more detailed examples and demos, check out the [examples page](https://www.docs.ivy.dev/demos/examples_and_demos.html).

  <details>
    <summary><b>Transpiling code between frameworks</b></summary>
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
    <summary><b>Tracing a computational graph</b></summary>
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

Ivy's transpiler allows you to use code from any other framework in your own code.
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

## Contributing

Contribute to Ivy and help shape the future of framework interoperability!  We welcome all contributions, from code to documentation to feedback.

Check out all of our [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html),
and find out more info in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html)
in the docs.

<br clear="all" />

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" alt="Contributors graph" />
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

Key improvements and SEO considerations:

*   **Headline:**  A strong, keyword-rich headline that captures the core value proposition:  "Ivy: Unleash Inter-Framework ML Code Conversion."
*   **One-Sentence Hook:**  A clear and concise introduction that immediately states Ivy's purpose.
*   **Keywords:** Incorporated relevant keywords like "machine learning," "framework conversion," "PyTorch," "TensorFlow," "JAX," "NumPy," "transpile," and "interoperability."
*   **Bulleted Key Features:** Highlights the main benefits in an easy-to-scan format.
*   **Clear Headings:**  Organized the README with clear, descriptive headings for better readability and SEO.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Emphasis on Benefits:**  Focused on what Ivy *does* for the user (e.g., "Unleash," "simplifies machine learning," "seamless conversion") rather than just stating features.
*   **Internal Linking:** Kept all internal links (docs, examples, etc.)
*   **Call to Action:** Encouraged contributions.
*   **Conciseness:** Removed redundant text and focused on essential information.
*   **Formatting:** Used Markdown formatting for improved readability.