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

# Gradio: Build AI Web Apps with Ease

**Gradio empowers you to create beautiful, shareable web applications for your machine learning models, APIs, and Python functions in minutes.** [Check out the original repository here!](https://github.com/gradio-app/gradio)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Key Features of Gradio:

*   **Rapid Prototyping:** Quickly build interactive demos and web apps without needing to know JavaScript, CSS, or web hosting.
*   **Simple Python API:** Use just a few lines of Python code to create user interfaces.
*   **Easy Sharing:** Share your demos with the world using a simple link, no server setup required.
*   **Flexible Components:** Choose from a wide range of pre-built UI components for various data types (text, images, audio, etc.).
*   **Customization Options:** Leverage `gr.Blocks` for advanced layouts and interactions.
*   **Chatbot Interface:** Create chatbot interfaces with `gr.ChatInterface`.
*   **Ecosystem Integration:** Seamlessly integrates with Hugging Face Spaces, Python and JavaScript clients for programmatic access, and Gradio-Lite for in-browser apps.

## Getting Started

### Installation

**Prerequisite**: Ensure you have [Python 3.10 or higher](https://www.python.org/downloads/) installed.

Install Gradio using pip:

```bash
pip install --upgrade gradio
```
> [!TIP]
> It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>. 

### Build Your First Demo
Create a simple Gradio app with the following code:

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

Run this code in your terminal (e.g., `python app.py`) to launch the demo in your browser (typically at [http://localhost:7860](http://localhost:7860)). 
Type your name, adjust the slider, and see the output.
![`hello_world_4` demo](demo/hello_world_4/screenshot.gif)

> [!TIP]
> When developing locally, you can run your Gradio app in <strong>hot reload mode</strong>, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in <code>gradio</code> before the name of the file instead of <code>python</code>. In the example above, you would type: `gradio app.py` in your terminal. You can also enable <strong>vibe mode</strong> by using the <code>--vibe</code> flag, e.g. <code>gradio --vibe app.py</code>, which provides an in-browser chat that can be used to write or edit your Gradio app using natural language. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

#### Understanding the `Interface` Class

The `gr.Interface` class is the core of Gradio. It wraps a Python function with a user interface. Key parameters:

*   `fn`: The Python function to wrap.
*   `inputs`: The Gradio components for input.
*   `outputs`: The Gradio components for output.

## Sharing Your Demo

Share your demo by adding `share=True` to the `launch()` call:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

This generates a public URL for your demo (e.g., `https://a23dsf231adb.gradio.live`) that anyone can access.

## Core Gradio Components

### Custom Demos with `gr.Blocks`

For custom layouts and data flows, use `gr.Blocks`. This allows for more complex interactions, including controlling component visibility and updating properties based on user input.

### Chatbots with `gr.ChatInterface`

Create chatbot UIs easily using `gr.ChatInterface`.

## Gradio Ecosystem

*   [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): Query Gradio apps programmatically in Python.
*   [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): Query Gradio apps programmatically in JavaScript.
*   [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): Write Gradio apps in Python that run entirely in the browser.
*   [Hugging Face Spaces](https://huggingface.co/spaces): Host Gradio applications for free.

## Next Steps

Explore the Gradio Guides for more detailed information and examples: [Gradio Guides](https://www.gradio.app/guides/)

For technical documentation, visit: [Gradio Docs](https://www.gradio.app/docs/)

## Gradio Sketch
You can also build Gradio applications without writing any code. Simply type `gradio sketch` into your terminal to open up an editor that lets you define and modify Gradio components, adjust their layouts, add events, all through a web editor. Or [use this hosted version of Gradio Sketch, running on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Get Involved

For bug reports or feature requests, submit an [issue on GitHub](https://github.com/gradio-app/gradio/issues/new/choose).  Join the [Discord server](https://discord.com/invite/feTf9x3ZSB) for general questions.

If you like Gradio, please leave a ⭐ on GitHub!

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

If you use Gradio in your work, please cite the following paper:

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```