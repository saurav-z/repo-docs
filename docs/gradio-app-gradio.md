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

# Gradio: Build & Share Machine Learning Demos - Fast!

**Gradio is the open-source Python library that allows you to create beautiful, interactive demos and web applications for your machine learning models with ease.**

[View the original repo on GitHub](https://github.com/gradio-app/gradio)

**Key Features:**

*   **Simple Setup:** Build demos with just a few lines of Python code.
*   **Interactive Components:** Includes a wide range of UI components (textboxes, images, sliders, etc.) for diverse ML applications.
*   **Easy Sharing:** Instantly share your demos with a public URL with the `share=True` parameter in `launch()`.
*   **Customizable Layouts:** Use `gr.Blocks` for fine-grained control over your app's layout and data flow.
*   **Chatbot Support:** Built-in `gr.ChatInterface` for creating chatbot interfaces quickly.
*   **Ecosystem:**  Leverage the `gradio_client`, `@gradio/client`, `@gradio/lite`, and Hugging Face Spaces for extended functionality.
*   **Hot Reloading:** Make changes to your Gradio app and see the results instantly with hot reload mode (`gradio app.py`)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Getting Started

### Installation

**Prerequisite**: Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
> It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

### Your First Gradio App

Create a simple demo:

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

Run the code (e.g., `python app.py`) and the demo will open in your browser.

![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

### Understanding the `Interface` Class

The `gr.Interface` class simplifies demo creation with three core arguments:

*   `fn`: The Python function your demo wraps.
*   `inputs`:  Gradio components for input (e.g., `"text"`, `gr.Slider()`).
*   `outputs`: Gradio components for output (e.g., `"textbox"`, `gr.Image()`).

### Sharing Your Demo

Share your demo with ease using the `share=True` parameter in `launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

### Beyond `Interface`

*   **`gr.Blocks`:** Create highly customized layouts and interactions.
*   **`gr.ChatInterface`:** Build chatbot interfaces quickly.
*   **Ecosystem:** Includes Python and JavaScript clients and Hugging Face Spaces integration.

## Additional Resources

*   [Gradio Guides](https://www.gradio.app/guides/)
*   [Technical API Documentation](https://www.gradio.app/docs/)
*   [Gradio Sketch](https://huggingface.co/spaces/aliabid94/Sketch)

## Get in Touch

*   [Report Issues on GitHub](https://github.com/gradio-app/gradio/issues/new/choose)
*   [Join our Discord Server](https://discord.com/invite/feTf9x3ZSB)

If you find Gradio helpful, please star it on GitHub!

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

Gradio is licensed under the Apache License 2.0 found in the [LICENSE](LICENSE) file in the root directory of this repository.

## Citation

Also check out the paper _[Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild](https://arxiv.org/abs/1906.02569), ICML HILL 2019_, and please cite it if you use Gradio in your work.

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}