<div align="center">
<a href="https://gradio.app">
<img src="readme_files/gradio.svg" alt="gradio" width=350>
</a>
</div>

<div align="center">
<span>
<a href="https://www.producthunt.com/posts/gradio-5-0?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-gradio&#0045;5&#0045;0" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=501906&theme=light" alt="Gradio&#0032;5&#0046;0 - the&#0032;easiest&#0032;way&#0032;to&#0032;build&#0032;AI&#0032;web&#0032;apps | Product Hunt" style="width: 150px; height: 54px;" width="150" height="54" /></a>
<a href="https://trendshift.io/repositories/2145" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2145" alt="gradio-app%2Fgradio | Trendshift" style="width: 150px; height: 55px;" width="150" height="55"/></a>
</span>

[![gradio-backend](https://github.com/gradio-app/gradio/actions/workflows/test-python.yml/badge.svg)](https://github.com/gradio-app/gradio/actions/workflows/test-python.yml)
[![gradio-ui](https://github.com/gradio-app/gradio/actions/workflows/tests-js.yml/badge.svg)](https://github.com/gradio-app/gradio/actions/workflows/tests-js.yml) 
[![PyPI](https://img.shields.io/pypi/v/gradio)](https://pypi.org/project/gradio/)
[![PyPI downloads](https://img.shields.io/pypi/dm/gradio)](https://pypi.org/project/gradio/)
![Python version](https://img.shields.io/badge/python-3.10+-important)
[![Twitter follow](https://img.shields.io/twitter/follow/gradio?style=social&label=follow)](https://twitter.com/gradio)

[Website](https://gradio.app)
| [Documentation](https://gradio.app/docs/)
| [Guides](https://gradio.app/guides/)
| [Getting Started](https://gradio.app/getting_started/)
| [Examples](demo/)

</div>

<div align="center">

English | [中文](readme_files/zh-cn#readme)

</div>

# Gradio: Build Machine Learning Web Apps with Ease

**Gradio empowers you to instantly create interactive web apps and demos for your machine learning models, APIs, or any Python function, all without needing to know HTML, CSS, or JavaScript.**  This README provides a summary and highlights the key features. For more details, visit the [Gradio GitHub repository](https://github.com/gradio-app/gradio).

**Key Features:**

*   🚀 **Rapid Prototyping:** Build demos and web applications in just a few lines of Python code.
*   💻 **No Frontend Skills Required:** Forget about JavaScript, CSS, or web hosting – Gradio handles the complexities.
*   🔗 **Easy Sharing:** Instantly share your demos with a public URL using Gradio's built-in sharing feature (with `share=True`).
*   🛠️ **Flexible Interface Class:** Create user interfaces for ML models with `gr.Interface` using `fn`, `inputs`, and `outputs` to define the UI.
*   🧱 **Customizable Blocks:** Use `gr.Blocks` for more advanced layouts and data flow control to tailor the application design.
*   💬 **Chatbot Creation:** Quickly build Chatbot UIs with the `gr.ChatInterface`.
*   🔌 **Python & JavaScript Ecosystem:** Leverage the `gradio` Python library, along with Python and JavaScript clients, to query any Gradio app programmatically.
*   🖼️ **Browser-Based Apps:** Develop Gradio apps that run entirely in the browser using `@gradio/lite`.
*   🎨 **Gradio Sketch:** Design and modify Gradio components without writing code using the web editor (`gradio sketch` command).

## Installation

To get started, install Gradio using pip:

```bash
pip install --upgrade gradio
```

## Basic Usage

Here's a simple "Hello, World!" example to illustrate the basics:

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
```

## Core Components

*   **`gr.Interface`**: Use this class for simple input-output applications.
*   **`gr.Blocks`**: For creating highly customized apps with multiple data flows, layouts, and dynamic updates.
*   **`gr.ChatInterface`**:  Build Chatbot applications.

## What's Next?

*   [Gradio Guides](https://www.gradio.app/guides/) for detailed instructions and example code.
*   [API documentation](https://www.gradio.app/docs/) for technical references.

## Additional Resources

*   [Hugging Face Spaces](https://huggingface.co/spaces) - Host your Gradio applications for free.
*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Query any Gradio app programmatically in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Query any Gradio app programmatically in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): Write Gradio apps in Python that run entirely in the browser (no server needed!), thanks to Pyodide.

## Questions & Support

*   Report bugs or suggest features via [GitHub issues](https://github.com/gradio-app/gradio/issues/new/choose).
*   Get help with usage on our [Discord server](https://discord.com/invite/feTf9x3ZSB).

## Open Source Stack

Gradio is built on many open-source libraries.  Please see the original README.

## License

Gradio is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

## Citation

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}