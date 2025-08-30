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

# Gradio: Build & Share Machine Learning Demos in Minutes

**Gradio is a powerful Python library that empowers you to build and share interactive web applications for your machine learning models, APIs, and any Python function with ease.** [Explore the Gradio repository](https://github.com/gradio-app/gradio) for more details.

## Key Features

*   **Rapid Prototyping:** Create interactive demos with minimal code – no front-end experience needed!
*   **Easy Sharing:** Generate shareable links with a single line of code to showcase your models.
*   **Flexible Components:** Utilize a wide range of UI components (text boxes, images, sliders, etc.) for diverse applications.
*   **Customization Options:** Design advanced web apps with custom layouts and data flows using `gr.Blocks`.
*   **Chatbot Creation:** Quickly build chatbot interfaces with the dedicated `gr.ChatInterface` class.
*   **Cross-Platform:** Works seamlessly in your code editor, Jupyter notebooks, Google Colab, and more.
*   **Full Ecosystem:** Leverage the `gradio_client`, `@gradio/client`, and `@gradio/lite` libraries to query Gradio apps programmatically in Python and JavaScript.
*   **Browser-Based Editing:** Easily create Gradio apps with `gradio sketch`, a web editor.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Installation

Ensure you have [Python 3.10 or higher](https://www.python.org/downloads/) installed. Install Gradio using pip:

```bash
pip install --upgrade gradio
```

>   [!TIP]
>   It's recommended to install Gradio within a virtual environment. Detailed installation instructions are available [here](https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment).

## Getting Started: Build Your First Demo

Create a basic demo in just a few lines of Python:

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

*   Run your code (e.g., `python app.py`) and access your demo at [http://localhost:7860](http://localhost:7860). Within a notebook, it'll be embedded directly.
    ![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

### Hot Reloading

You can run Gradio apps in hot reload mode, which automatically updates your app as you save changes.

To do this, type `gradio` before the file name: `gradio app.py`. You can also use the `--vibe` flag for an in-browser chat that helps to write or edit your app. Read more in the [Hot Reloading Guide](https://www.gradio.app/guides/developing-faster-with-reload-mode).

### Understanding the `Interface` Class

The `gr.Interface` class is central to Gradio, enabling you to build demos rapidly:

-   `fn`: Your Python function to wrap.
-   `inputs`: Gradio components for input (e.g., "textbox", `gr.Image()`).
-   `outputs`: Gradio components for output.

>   [!TIP]
>   You can pass component names as strings (e.g., `"textbox"`) or component class instances (e.g., `gr.Textbox()`) to `inputs` and `outputs`.

Explore the [building Interfaces guide](https://www.gradio.app/guides/the-interface-class) for deeper insights.

## Sharing Your Demo

Make your demo accessible to anyone, anywhere:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

Set `share=True` in `launch()` to generate a public URL (e.g., `https://a23dsf231adb.gradio.live`). Learn more about sharing in the [dedicated guide](https://www.gradio.app/guides/sharing-your-app).

## Diving Deeper into Gradio

Beyond `Interface`, Gradio offers powerful features:

### Custom Demos with `gr.Blocks`

Design intricate web apps with `gr.Blocks`, gaining control over layouts and data flows. See examples like the [Automatic1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for inspiration. Learn more in the [building with Blocks guide](https://www.gradio.app/guides/blocks-and-event-listeners).

### Chatbots with `gr.ChatInterface`

Rapidly build chatbot interfaces using `gr.ChatInterface`. Consult our [guide on `gr.ChatInterface`](https://www.gradio.app/guides/creating-a-chatbot-fast).

### Gradio Ecosystem

*   **Gradio Python Client** (`gradio_client`): Query Gradio apps in Python.
*   **Gradio JavaScript Client** (`@gradio/client`): Query Gradio apps in JavaScript.
*   **Gradio-Lite** (`@gradio/lite`): Run Gradio apps entirely in the browser.
*   **Hugging Face Spaces**: Host Gradio applications for free.

## Next Steps

*   Follow the Gradio Guides: Learn sequentially through Gradio Guides, including examples and interactive demos. Start with the [Interface class guide](https://www.gradio.app/guides/the-interface-class).
*   Consult the [technical API documentation](https://www.gradio.app/docs/) for details.

### Gradio Sketch

Build Gradio apps without code using `gradio sketch` in your terminal, or try the hosted version on [Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Get Support

*   Report bugs or request features via [GitHub issues](https://github.com/gradio-app/gradio/issues/new/choose).
*   Ask questions on our [Discord server](https://discord.com/invite/feTf9x3ZSB).

If you like Gradio, please leave us a ⭐ on GitHub!

## Open Source Stack

Gradio is built on amazing open-source libraries!

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

Gradio is licensed under the Apache License 2.0; find the [LICENSE](LICENSE) in the repository's root.

## Citation

If you use Gradio in your work, please cite the paper:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```