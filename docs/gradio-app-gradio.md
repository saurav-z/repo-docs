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

# Gradio: Build and Share Machine Learning Web Apps in Minutes

**Gradio is the fastest way to create user-friendly web interfaces for your machine learning models, APIs, or any Python function, and share them with anyone, anywhere.** [Explore the Gradio repo on GitHub](https://github.com/gradio-app/gradio).

*   **Fast Prototyping:** Quickly build interactive demos for your ML models with minimal code.
*   **No Frontend Expertise Required:** Design web apps without needing to know JavaScript, CSS, or web hosting.
*   **Easy Sharing:** Share your demos with a public URL in seconds with the `share=True` parameter.
*   **Flexible & Customizable:** Offers high-level and low-level approaches with `Interface`, `Blocks`, and `ChatInterface` classes.
*   **Rich Component Library:** Includes over 30 built-in components for diverse ML applications (text, images, audio, etc.).
*   **Seamless Integration:** Works with popular platforms like Hugging Face Spaces.
*   **Comprehensive Ecosystem:** Includes Python and JavaScript clients for programmatic access, and in-browser app development with Gradio-Lite.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Getting Started

### Installation

**Prerequisite**: Ensure you have [Python 3.10 or higher](https://www.python.org/downloads/) installed.

Install Gradio using pip:

```bash
pip install --upgrade gradio
```

> [!TIP]
> Install Gradio in a virtual environment for best practices. Detailed instructions are available [here](https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment).

### Build Your First Demo

Create an interactive demo in just a few lines of Python.

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
> Use `gr` as an alias for `gradio` for cleaner code.

Run the Python code in your terminal or code editor. Access the demo in your browser at `http://localhost:7860`. If you are running within a notebook, the demo will appear embedded within the notebook.

![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

Enter your name in the textbox and adjust the slider to see the demo in action.

> [!TIP]
> Use <strong>hot reload mode</strong> for local development by typing <code>gradio</code> before the filename (e.g., <code>gradio app.py</code>).  You can also enable <strong>vibe mode</strong> using the <code>--vibe</code> flag: `gradio --vibe app.py`.

### Understanding the `Interface` Class

The `gr.Interface` class simplifies demo creation with these core arguments:

*   `fn`:  The Python function to wrap.
*   `inputs`:  Gradio components for input (matching function arguments).
*   `outputs`: Gradio components for output (matching function return values).

Gradio offers diverse [components](https://www.gradio.app/docs/gradio/introduction) to fit various ML tasks.  Pass component names as strings or instances (`gr.Textbox()`).

### Sharing Your Demo

Share your demo globally by adding `share=True` in the `launch()` function:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

demo.launch(share=True)
```

A public URL (e.g., `https://a23dsf231adb.gradio.live`) is generated. Find out more in the [sharing guide](https://www.gradio.app/guides/sharing-your-app).

## Gradio Overview

### Custom Demos with `gr.Blocks`

For custom layouts and complex data flows, use `gr.Blocks` to have granular control over web app design and component interactions. Build advanced applications like the [Automatic1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).  Learn more in the [building with Blocks guide](https://www.gradio.app/guides/blocks-and-event-listeners).

### Chatbots with `gr.ChatInterface`

The `gr.ChatInterface` class is tailored for creating chatbot UIs. For chatbot development, see the [gr.ChatInterface guide](https://www.gradio.app/guides/creating-a-chatbot-fast).

### Gradio Ecosystem

Gradio is more than just a Python library! It's a whole ecosystem including:

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`)
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`)
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`)
*   [Hugging Face Spaces](https://huggingface.co/spaces)

## Next Steps

*   Explore the [Gradio Guides](https://www.gradio.app/guides/) for detailed learning and example code.
*   Consult the [technical API documentation](https://www.gradio.app/docs/) for in-depth information.

### Gradio Sketch

Build apps visually with [Gradio Sketch](https://huggingface.co/spaces/aliabid94/Sketch) without writing code, by typing `gradio sketch` into your terminal.

## Get Help

*   Report bugs or request features by opening an [issue on GitHub](https://github.com/gradio-app/gradio/issues/new/choose).
*   Get general usage support on [our Discord server](https://discord.com/invite/feTf9x3ZSB).

If you like Gradio, please leave us a ⭐ on GitHub!

## Open Source Stack

Gradio leverages these open-source libraries:

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

Gradio is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file.

## Citation

Cite this work if you use Gradio:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```