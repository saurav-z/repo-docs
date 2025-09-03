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

# Gradio: Build and Share ML Apps with Ease

**Gradio is the open-source Python library that empowers you to rapidly build and share interactive web applications for your machine learning models, APIs, and any Python function, without needing any front-end experience.**  ([See the original repo](https://github.com/gradio-app/gradio))

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

**Key Features:**

*   **Rapid Prototyping:** Quickly create demos and web apps in just a few lines of Python code.
*   **Intuitive Interface:**  Easy-to-use components for inputs and outputs (text boxes, images, sliders, etc.).
*   **Effortless Sharing:**  Share your demos with a public URL using the `share=True` parameter.
*   **No Front-End Required:** Eliminate the need for JavaScript, CSS, or web hosting expertise.
*   **Customization Options:** Build complex UIs with `gr.Blocks` for flexible layouts and data flows.
*   **Chatbot Creation:** Easily create chatbot interfaces with `gr.ChatInterface`.
*   **Comprehensive Ecosystem:** Integrates with Python and JavaScript clients, and supports browser-based execution via Gradio-Lite.
*   **Hot Reloading:** Run Gradio apps in hot reload mode for faster development.

## Installation

**Prerequisite:** Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

## Building Your First Demo

Here's a simple "Hello World" example:

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

Run this code (e.g., `python app.py`), and a demo will open in your browser.  You can input text, use the slider, and see the output.

> [!TIP]
 > We shorten the imported name from <code>gradio</code> to <code>gr</code>. This is a widely adopted convention for better readability of code. 

### Understanding the `Interface` Class

The `gr.Interface` class is the core building block for creating demos:

*   `fn`:  Your Python function to wrap.
*   `inputs`: Gradio components for input (text, images, sliders, etc.).
*   `outputs`: Gradio components for output.

## Sharing Your Demo

Easily share your demo with the world:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

The `share=True` parameter generates a public URL (e.g., `https://a23dsf231adb.gradio.live`).

## Additional Information

*   **Custom Demos with `gr.Blocks`**: For more control over layouts and data flow.
*   **Chatbots with `gr.ChatInterface`**: For quickly creating chatbot UIs.
*   **Gradio Ecosystem**: Explore Python & JavaScript clients (`gradio_client`, `@gradio/client`), browser-based execution via `@gradio/lite`, and Hugging Face Spaces integration.
*   **Gradio Sketch**: Build Gradio apps without code using a web editor (type `gradio sketch` or use the [hosted version](https://huggingface.co/spaces/aliabid94/Sketch)).

## Get Started and Learn More

*   **Gradio Guides:** [Explore the Guides](https://www.gradio.app/guides/) to learn more.
*   **API Documentation:** [Access the API Documentation](https://www.gradio.app/docs/)
*   **Open Source Stack:** Gradio is built on a range of wonderful open-source libraries.

## Questions and Support

Report issues on [GitHub](https://github.com/gradio-app/gradio/issues/new/choose) and ask general questions on [Discord](https://discord.com/invite/feTf9x3ZSB).

## Citation

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}