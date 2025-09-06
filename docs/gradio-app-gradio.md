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

# Gradio: Build Machine Learning Web Apps in Python

**Gradio empowers you to create interactive web applications for your machine learning models and data science projects in minutes, with no web development experience required.**  Check out the [Gradio GitHub Repository](https://github.com/gradio-app/gradio) for the source code.

Key Features:

*   **Rapid Prototyping:** Quickly build interactive demos and web apps for your machine learning models.
*   **Intuitive Python API:** Create user interfaces with just a few lines of Python code.
*   **No Web Development Required:**  Focus on your model – Gradio handles the front-end.
*   **Shareable Demos:** Easily share your demos with others via public URLs.
*   **Rich Component Library:**  Supports over 30 built-in components for various input and output types.
*   **Customizable Layouts:**  Use `gr.Blocks` for advanced UI design and control.
*   **Chatbot Support:**  Simplified chatbot creation with `gr.ChatInterface`.
*   **Ecosystem of Tools:** Includes Python and JavaScript clients for programmatic access, and integration with Hugging Face Spaces.

## Getting Started

### Installation

**Prerequisite**:  Requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

### Build Your First Demo

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

*   Run your Python code (e.g., `python app.py`).
*   The demo opens in your browser (e.g., `http://localhost:7860`) or embedded in a notebook.
*   Interact with the demo by typing your name, adjusting the slider, and submitting.

> [!TIP]
 > When developing locally, you can run your Gradio app in <strong>hot reload mode</strong>, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in <code>gradio</code> before the name of the file instead of <code>python</code>. In the example above, you would type: `gradio app.py` in your terminal. You can also enable <strong>vibe mode</strong> by using the <code>--vibe</code> flag, e.g. <code>gradio --vibe app.py</code>, which provides an in-browser chat that can be used to write or edit your Gradio app using natural language. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

### Key Classes

*   **`gr.Interface`:** Simplifies demo creation with `fn`, `inputs`, and `outputs`.
*   **`gr.Blocks`:**  For more advanced UI customization and control.
*   **`gr.ChatInterface`:** Build chatbots with ease.

### Sharing Your Demo

Share your demo with the world by setting `share=True` in `demo.launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

This generates a public URL (e.g., `https://a23dsf231adb.gradio.live`) for your demo.

## Explore Gradio's Ecosystem

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Query Gradio apps in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Query Gradio apps in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): In-browser Gradio apps using Pyodide.
*   [Hugging Face Spaces](https://huggingface.co/spaces): Host your Gradio apps for free.

## Next Steps

*   [Gradio Guides](https://www.gradio.app/guides/): Comprehensive tutorials.
*   [API Documentation](https://www.gradio.app/docs/): Technical reference.
*   [Gradio Sketch](https://huggingface.co/spaces/aliabid94/Sketch): Build Gradio apps visually without code.

## Get Help

*   [GitHub Issues](https://github.com/gradio-app/gradio/issues/new/choose): Report bugs and feature requests.
*   [Discord Server](https://discord.com/invite/feTf9x3ZSB): Get help with usage questions.

## Open Source Stack

[Insert logos from original README here]

## License

Apache License 2.0 ([LICENSE](LICENSE))

## Citation

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```