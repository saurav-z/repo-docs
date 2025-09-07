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

# Gradio: The Fastest Way to Build & Share AI Web Apps (in Python)

Gradio is an open-source Python library that empowers you to effortlessly build, demo, and share interactive web applications for your machine learning models and data science projects.  <a href="https://github.com/gradio-app/gradio">Check out the original repo on GitHub</a>.

**Key Features:**

*   **Rapid Prototyping:** Quickly create user-friendly interfaces for your models with minimal code.
*   **No Frontend Experience Required:** Build web apps without writing any JavaScript, CSS, or dealing with web hosting.
*   **Share with Ease:** Generate shareable links to your demos in seconds, perfect for collaboration and showcasing your work.
*   **Rich Component Library:** Utilize over 30 built-in components to create diverse and engaging user interfaces.
*   **Customization Options:** Leverage `gr.Blocks` for complete layout control and advanced interactions.
*   **Chatbot Creation:** Easily build chatbot interfaces with `gr.ChatInterface`.
*   **Python & JavaScript Ecosystem:** Access a comprehensive ecosystem including Python and JavaScript clients for programmatic access to Gradio apps and integrations with other services like Hugging Face Spaces.
*   **In-Browser Editor:** Rapidly prototype Gradio applications using Gradio Sketch.

## Installation

**Prerequisites:** Python 3.10 or higher is required.

Install Gradio using pip:

```bash
pip install --upgrade gradio
```

> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>. 

## Building Your First Demo

Create a Gradio app in just a few lines of Python. Here's a simple example:

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

Run the code, and your demo will open in your browser (usually at `http://localhost:7860`).

![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

## Understanding the `Interface` Class

The `gr.Interface` class is the core of Gradio for creating demos.  It takes three main arguments:

*   `fn`: Your Python function.
*   `inputs`:  The input components (e.g., text boxes, sliders).
*   `outputs`: The output components.

> [!TIP]
 > For the `inputs` and `outputs` arguments, you can pass in the name of these components as a string (`"textbox"`) or an instance of the class (`gr.Textbox()`).

## Sharing Your Demo

To share your demo, set `share=True` in the `launch()` method:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

Gradio will generate a public URL (e.g., `https://a23dsf231adb.gradio.live`) for your demo.

## Diving Deeper into Gradio

Beyond `Interface`, Gradio offers:

*   **`gr.Blocks`:** For fully customizable layouts and data flows.  Build complex applications.
*   **`gr.ChatInterface`:** Specifically designed for chatbot UIs.
*   **Gradio Ecosystem:** Includes Python and JavaScript clients for interacting with Gradio apps programmatically, as well as integration with Hugging Face Spaces.

## What's Next?

Explore the [Gradio Guides](https://www.gradio.app/guides/) for in-depth tutorials.

For technical details, refer to the [API documentation](https://www.gradio.app/docs/).

## Additional Resources

*   **Gradio Sketch:** Build Gradio apps without code using the web editor or [use this hosted version of Gradio Sketch, running on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Get Involved

*   **Report Issues and Feature Requests:**  Create an [issue on GitHub](https://github.com/gradio-app/gradio/issues/new/choose).
*   **Ask Questions:** Join our [Discord server](https://discord.com/invite/feTf9x3ZSB).
*   **Star Us:** Show your support by starring the repo on GitHub!

## Open Source Stack

Gradio is built with a range of awesome open-source libraries:

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

## License & Citation

Gradio is licensed under the Apache License 2.0 ([LICENSE](LICENSE)).

Cite Gradio:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}