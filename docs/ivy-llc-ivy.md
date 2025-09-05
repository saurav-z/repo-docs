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
    <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/test-transpiler.yml/badge.svg" alt="Test Transpiler"/>
  </a>
  <a href="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml">
    <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests"/>
  </a>
</div>
<br clear="all" />

## Ivy: Effortlessly Convert ML Code Between Frameworks

**Ivy allows you to seamlessly convert machine learning code between different frameworks like PyTorch, TensorFlow, JAX, and NumPy with ease.**

### Key Features:

*   **Framework Conversion:** Transpile code between PyTorch, TensorFlow, JAX, and NumPy.
*   **Easy Installation:** Install quickly using `pip install ivy`.
*   **Graph Tracing:** Trace computational graphs for efficient execution.
*   **Eager and Lazy Transpilation:** Supports eager transpilation for functions/classes and lazy transpilation for modules.
*   **Comprehensive Documentation:** Extensive documentation with examples and demos available at [https://www.docs.ivy.dev/](https://www.docs.ivy.dev/).

### Supported Frameworks:

Ivy currently supports conversions between the following frameworks:

| Framework     | Source | Target |
| ------------- | :----: | :----: |
| PyTorch       |   ✅   |   🚧   |
| TensorFlow    |   🚧   |   ✅   |
| JAX           |   🚧   |   ✅   |
| NumPy         |   🚧   |   ✅   |

### Getting Started

Install Ivy with:

```bash
pip install ivy
```

### Example Usage

*   **Transpiling Code:**

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

*   **Tracing a Computational Graph:**

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

### Learn More

*   **API Documentation:** Explore the full API reference for functions like `ivy.transpile()` and `ivy.trace_graph()`.
*   **Demos and Tutorials:** Discover more use cases and examples on the [examples page](https://www.docs.ivy.dev/demos/examples_and_demos.html).

### Contributing

We welcome contributions! Learn more about contributing and find open tasks in our [Contributing Guide](https://docs.ivy.dev/overview/contributing.html) and [Open Tasks](https://docs.ivy.dev/overview/contributing/open_tasks.html).

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

**[View the original repository on GitHub](https://github.com/ivy-llc/ivy)**