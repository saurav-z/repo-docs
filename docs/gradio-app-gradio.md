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

**Quickly create and share interactive web applications for your machine learning models using Python with Gradio.**

Gradio is an open-source Python library designed to streamline the process of building and sharing demos or web applications for your machine learning models, APIs, or any Python function. With Gradio, you can build and share your interactive demos in seconds. Forget about complex web development – Gradio makes it simple!

**Key Features:**

*   **Rapid Prototyping:** Build interactive demos in just a few lines of Python code.
*   **No Web Development Required:** Eliminates the need for JavaScript, CSS, or web hosting experience.
*   **Shareable Demos:** Easily share your demos with a public URL using the `share=True` parameter.
*   **Rich Component Library:** Offers over [30 built-in components](https://www.gradio.app/docs/gradio/introduction) to create engaging interfaces.
*   **Flexible Interface Class:** Uses the `Interface` class for straightforward demo creation with `fn`, `inputs`, and `outputs` arguments.
*   **Customizable Layouts:** Utilize `gr.Blocks` for more complex and customized layouts and data flows.
*   **Chatbot Creation:** Use `gr.ChatInterface` to create chatbot UIs.
*   **Ecosystem of Tools:** Includes Python and JavaScript clients for programmatic interaction, along with browser-based app development tools (Gradio-Lite) and hosting platforms.
*   **Hot Reloading & Vibe Mode:** Develop faster with automatic reloading and an in-browser chat for code editing.
*   **Gradio Sketch:** Build Gradio applications without writing any code, using a visual web editor.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Installation

**Prerequisite:** Requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
> It's recommended to install Gradio within a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

## Getting Started: Build Your First Demo

Create your first Gradio app using your preferred code editor, Jupyter notebook, Google Colab, or anywhere you write Python:

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
> Using `gr` for `gradio` improves code readability.

Run the code (e.g., `python app.py`). The demo will open in your browser (usually at `http://localhost:7860`).

![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

Enter your name and adjust the slider, then click Submit.

> [!TIP]
> Use <code>gradio app.py</code> in your terminal to run in <strong>hot reload mode</strong>. Enable <strong>vibe mode</strong> using <code>--vibe</code> to write or edit your Gradio app using natural language.

## Understanding the `Interface` Class

The `gr.Interface` class generates demos for ML models with inputs and outputs. Core arguments:

*   `fn`: The function to wrap.
*   `inputs`: Input Gradio component(s).
*   `outputs`: Output Gradio component(s).

You can use strings (e.g., `"textbox"`) or class instances (e.g., `gr.Textbox()`) for `inputs` and `outputs`.

For multiple inputs/outputs, pass lists of components. Learn more in the [building Interfaces guide](https://www.gradio.app/main/guides/the-interface-class).

## Sharing Your Demo

To share your demo, set `share=True` in `launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

This generates a public URL (e.g., `https://a23dsf231adb.gradio.live`) for anyone to access your demo.  Read the guide on [sharing your Gradio application](https://www.gradio.app/guides/sharing-your-app) for more details.

## Diving Deeper into Gradio

*   **`gr.Blocks`:** Enables customizable layouts and data flows for more complex apps.
*   **`gr.ChatInterface`:** Simplifies the creation of chatbot UIs.
*   **Gradio Ecosystem:** Includes Python & JavaScript clients, Gradio-Lite, and integration with Hugging Face Spaces, allowing you to build machine learning applications in Python or JavaScript.

### Gradio Python & JavaScript Ecosystem

That's the gist of the core `gradio` Python library, but Gradio is actually so much more! It's an entire ecosystem of Python and JavaScript libraries that let you build machine learning applications, or query them programmatically, in Python or JavaScript. Here are other related parts of the Gradio ecosystem:

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): query any Gradio app programmatically in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): query any Gradio app programmatically in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): write Gradio apps in Python that run entirely in the browser (no server needed!), thanks to Pyodide.
*   [Hugging Face Spaces](https://huggingface.co/spaces): the most popular place to host Gradio applications — for free!

## What's Next?

Follow the Gradio Guides for sequential learning: [dive deeper into the Interface class](https://www.gradio.app/guides/the-interface-class). Explore the [technical API documentation](https://www.gradio.app/docs/) for more details.

### Gradio Sketch

Build Gradio applications visually with `gradio sketch` in your terminal or use the [hosted version on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Questions?

*   Report bugs/request features on [GitHub](https://github.com/gradio-app/gradio/issues/new/choose).
*   Get general usage help on [Discord](https://discord.com/invite/feTf9x3ZSB).

If you like Gradio, please give us a ⭐ on GitHub!

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

Gradio is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

## Citation

Please cite this paper if you use Gradio:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}