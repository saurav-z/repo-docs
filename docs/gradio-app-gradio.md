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

# Gradio: Build & Share Beautiful Machine Learning Web Apps in Minutes

Gradio is the open-source Python library that allows you to rapidly build and deploy user-friendly web interfaces for your machine learning models, APIs, and any Python function, making them accessible to anyone with a web browser.  [Explore the Gradio Repository](https://github.com/gradio-app/gradio).

**Key Features:**

*   **Effortless Web App Creation:** Create interactive demos with just a few lines of Python code. No need for JavaScript, CSS, or web hosting expertise.
*   **Fast Sharing:** Easily share your demos with the world using Gradio's built-in sharing capabilities. Generate a public URL with a simple `share=True` parameter.
*   **Rich Component Library:** Utilize a wide range of over 30 pre-built components (e.g., textboxes, images, audio players) designed for machine learning applications.
*   **Customizable Layouts:**  Design sophisticated web apps using `gr.Blocks` for complete control over layout and data flows.
*   **Chatbot Development:** Quickly build chatbot interfaces with `gr.ChatInterface`.
*   **Python & JavaScript Ecosystem:** Leverage the broader Gradio ecosystem, including Python and JavaScript clients for programmatic interaction, and tools like Gradio-Lite for in-browser app creation.
*   **Hot Reloading:**  Speed up development with hot reloading to see changes instantly.

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

### Build Your First Demo

Here's a simple "Hello, World!" example:

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

*   Run the code from your terminal (e.g., `python app.py`).
*   Access your demo in your web browser at the provided local URL (e.g., `http://localhost:7860`).
*   The demo will also appear embedded within the notebook if you run the code within a notebook.

> [!TIP]
 > Develop faster with Gradio's hot reload mode by typing `gradio` before the file name in your terminal (e.g., `gradio app.py`).  Explore `vibe mode` with `--vibe` for natural language-based app editing. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

### Understanding the `Interface` Class

The `gr.Interface` class is central to building simple demos:

*   `fn`: Your Python function.
*   `inputs`: Gradio input components.
*   `outputs`: Gradio output components.

> [!TIP]
 > You can specify input and output components as strings (e.g., `"textbox"`) or as instances of the class (e.g., `gr.Textbox()`).

### Sharing Your Demo

Share your demo with `share=True` in `launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

Gradio generates a public URL (e.g., `https://a23dsf231adb.gradio.live`), allowing anyone to access your demo.

Learn more in our dedicated guide on [sharing your Gradio application](https://www.gradio.app/guides/sharing-your-app).

## Gradio's Capabilities

### Custom Demos with `gr.Blocks`

For more complex layouts and data flows, use the `gr.Blocks` class. It offers customization, including controlling component placement, handling multiple data flows, and updating component properties based on user interaction.

We dive deeper into the `gr.Blocks` on our series on [building with Blocks](https://www.gradio.app/guides/blocks-and-event-listeners).

### Chatbots with `gr.ChatInterface`

Build chatbot UIs easily using `gr.ChatInterface`.

### The Gradio Ecosystem

Gradio offers a rich ecosystem:

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Interact with Gradio apps in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Interact with Gradio apps in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): Create in-browser Gradio apps with Pyodide.
*   [Hugging Face Spaces](https://huggingface.co/spaces): Host Gradio applications for free.

## What's Next?

*   Explore the [Gradio Guides](https://www.gradio.app/guides/) for step-by-step learning.
*   Consult the [API documentation](https://www.gradio.app/docs/) for technical details.

### Gradio Sketch

Build Gradio apps without coding using `gradio sketch` in your terminal, or use the [hosted version](https://huggingface.co/spaces/aliabid94/Sketch) on Hugging Face Spaces.

## Support

*   Report bugs or request features via [GitHub issues](https://github.com/gradio-app/gradio/issues/new/choose).
*   Get help on our [Discord server](https://discord.com/invite/feTf9x3ZSB).
*   Show your support with a ⭐ on GitHub!

## Open Source Stack

Gradio is built on open-source libraries:

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

Gradio is licensed under the Apache License 2.0, found in the [LICENSE](LICENSE) file.

## Citation

If you use Gradio, please cite the following paper:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```