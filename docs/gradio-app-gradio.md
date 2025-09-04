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

# Gradio: Build and Share Machine Learning Demos with Ease

**Gradio is a powerful, open-source Python library that allows you to rapidly create and share interactive web applications for your machine learning models and data, all without writing any HTML, CSS, or JavaScript.**  Visit the [Gradio GitHub](https://github.com/gradio-app/gradio) to learn more.

## Key Features

*   **Rapid Prototyping:** Quickly build demos for your models or APIs in just a few lines of Python.
*   **No Web Development Required:**  Skip the front-end hassle; Gradio handles the UI.
*   **Interactive Components:** Support for diverse input and output components including text boxes, images, audio, video, and more.
*   **Easy Sharing:**  Share your demos with a public URL with a simple `share=True` parameter in the `launch()` function.
*   **Customizable Layouts:**  Utilize `gr.Blocks` for full control over app design and data flow.
*   **Chatbot Interface:**  Quickly create chatbots using `gr.ChatInterface`.
*   **Hot Reloading & Vibe Mode**:  Enjoy hot reloading during development and vibe mode for interactive editing.
*   **Python & JavaScript Ecosystem:** Utilize Gradio's client libraries to query apps programmatically in Python or JavaScript.
*   **Gradio Sketch**: Build apps with a web editor, no code required.

## Installation

**Prerequisite:**  Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using pip:

```bash
pip install --upgrade gradio
```

> [!TIP]
 >  It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>. 

## Build Your First Demo

Create a simple "Hello, World!" demo with the following code:

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
> [!TIP]
 > We shorten the imported name from <code>gradio</code> to <code>gr</code>. This is a widely adopted convention for better readability of code. 

Run the code (e.g., `python app.py`) and access the demo in your browser (usually at `http://localhost:7860`).

**Understanding the `Interface` Class**

The `gr.Interface` class simplifies demo creation. It takes:

*   `fn`: The Python function to wrap.
*   `inputs`:  Gradio components for input.
*   `outputs`: Gradio components for output.

## Sharing Your Demo

Make your demo public with the `share=True` parameter in `launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo 🚀
```

This generates a shareable URL (e.g., `https://a23dsf231adb.gradio.live`).

## Gradio's Core Concepts

*   **`gr.Blocks`**: For building custom and complex web apps with more customizable layouts and data flows.
*   **`gr.ChatInterface`**:  For quickly creating chatbot UIs.

## The Gradio Ecosystem

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Programmatically query Gradio apps in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Programmatically query Gradio apps in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): Run Gradio apps entirely in the browser.
*   [Hugging Face Spaces](https://huggingface.co/spaces): Host Gradio applications for free.

## What's Next?

Explore the [Gradio Guides](https://www.gradio.app/guides/) for in-depth tutorials and the [API Documentation](https://www.gradio.app/docs/) for detailed information.

## Gradio Sketch

Build Gradio applications without coding using the `gradio sketch` command or the hosted version on [Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Get Help

*   [GitHub Issues](https://github.com/gradio-app/gradio/issues/new/choose) for bug reports and feature requests.
*   [Discord Server](https://discord.com/invite/feTf9x3ZSB) for general usage questions.

If you like Gradio, please leave us a ⭐ on GitHub!

## Open Source Stack

Gradio is built on top of many wonderful open-source libraries!

[<img src="readme_files/huggingface_mini.svg" alt="huggingface" height=40>](https://huggingface.co)
[<img src="readme_files/python.svg" alt="python" height=40>](https://www.python.org)
[<img src="readme_files/fastapi.svg" alt="fastapi" height=40>](https://fastapi.tiangolo.com)
[<img src="readme_files/encode.svg" alt="encode" height=40>](https://www.encode.io)
[<img src="readme_files/svelte.svg" alt="svelte" height=40>](https://svelte.dev)
[<img src="readme_files/vite.svg" alt="vite" height=40>](https://vitejs.dev)
[<img src="readme_files/pnpm.svg" alt="pnpm" height=40>](https://pnpm.io)
[<img src="readme_files/tailwind.svg" alt="tailwind" height=40>](https://tailwindcss.com)
[<img src="readme_files/storybook.svg" alt="storybook" height=40>](https://storybook.js.org/)
[<img src="readme_files/chromatic.svg" alt="chromatic" height=40>](https://www.chromatic.com/)

## License

Gradio is licensed under the Apache License 2.0 found in the [LICENSE](LICENSE) file.

## Citation

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}