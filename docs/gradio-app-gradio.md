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

**Gradio is the open-source Python library that lets you rapidly build and share beautiful, user-friendly web applications and demos for your machine learning models and data science projects.**

[View the original repository on GitHub](https://github.com/gradio-app/gradio)

**Key Features:**

*   **Rapid Prototyping:** Build interactive demos and web apps for your ML models in minutes, not months, using just a few lines of Python.
*   **No Frontend Expertise Required:** Eliminate the need for JavaScript, CSS, or web hosting knowledge.
*   **Easy Sharing:** Share your demos with a public URL in seconds for global accessibility.
*   **Versatile Components:** Utilize over 30 pre-built Gradio components to handle various data types like images, text, audio, and more.
*   **Customization Options:** Leverage `gr.Blocks` for complete control over layout and data flow and `gr.ChatInterface` for streamlined chatbot creation.
*   **Hot Reloading & Vibe Mode:** Develop faster with automatic reloading and a chat interface for app editing.
*   **Complete Ecosystem:** Explore the extended Gradio ecosystem with Python and JavaScript clients for programmatic access, browser-based execution via Gradio-Lite, and Hugging Face Spaces for hosting.

## Installation

**Prerequisite**: Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

## Building Your First Demo

Create a Gradio app quickly:

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

Run this code and access your demo in a browser.

![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

## Understanding the `Interface` Class

The `gr.Interface` class creates demos for machine learning models:

*   `fn`:  The Python function to wrap with a UI.
*   `inputs`: The Gradio component(s) for input.
*   `outputs`: The Gradio component(s) for output.

## Sharing Your Demo

Share your demo with the `share=True` parameter in `launch()`:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

This generates a public URL for your demo, enabling worldwide access.  Read more on [sharing your Gradio application](https://www.gradio.app/guides/sharing-your-app).

## An Overview of Gradio

Besides `Interface`, Gradio includes:

*   **Custom Demos with `gr.Blocks`:** For building web apps with customized layouts, data flows, and interactions, like the [Automatic1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Learn more on [building with Blocks](https://www.gradio.app/guides/blocks-and-event-listeners).
*   **Chatbots with `gr.ChatInterface`:** For creating chatbot UIs. You can jump straight to [our dedicated guide on `gr.ChatInterface`](https://www.gradio.app/guides/creating-a-chatbot-fast).
*   **Gradio Ecosystem:** Gradio extends to a comprehensive ecosystem:
    *   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Query Gradio apps programmatically in Python.
    *   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Query Gradio apps programmatically in JavaScript.
    *   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): Write Gradio apps in Python that run entirely in the browser.
    *   [Hugging Face Spaces](https://huggingface.co/spaces): Host Gradio applications.

## Next Steps

Explore the Gradio Guides for a step-by-step learning experience: [let's dive deeper into the Interface class](https://www.gradio.app/guides/the-interface-class).

Access the [technical API documentation](https://www.gradio.app/docs/) for specifics.

## Gradio Sketch

Build Gradio apps without writing code using the `gradio sketch` command, or use the [hosted version of Gradio Sketch, running on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Questions?

For bug reports and feature requests, use the [GitHub issues](https://github.com/gradio-app/gradio/issues/new/choose).  For general questions, join our [Discord server](https://discord.com/invite/feTf9x3ZSB).

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

Gradio is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

## Citation

Cite the paper _[Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild](https://arxiv.org/abs/1906.02569), ICML HILL 2019_:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```